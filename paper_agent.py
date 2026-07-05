import requests
import json
import time
import os
import re
from datetime import datetime

# ========== 可配置参数区 ==========
# 核心领域关键词（命中越多相关性越高）
DOMAIN_KEYWORDS = {
    "lidar": 3, "point cloud": 3, "slam": 3, "lio": 3, "loam": 3,
    "point cloud registration": 4, "loop closure": 4, "imu fusion": 3,
    "livox": 3, "gaussian splatting slam": 4, "nerf slam": 4,
    "semantic segmentation": 2, "autonomous driving": 2, "localization": 2,
    "odometry": 3, "feature extraction": 2, "voxel": 2
}
# 优质机构/顶会关键词（命中额外加分）
BONUS_KEYWORDS = [
    "ICRA", "IROS", "CVPR", "ECCV", "3DV", "RSS", "TRO",
    "CMU", "MIT", "Stanford", "Oxford", "HKUST", "Tsinghua",
    "Waymo", "Tesla", "Huawei", "Baidu", "DJI", "RoboSense", "Hesai"
]
# 代码开源相关关键词（本地快筛用）
CODE_KEYWORDS = [
    "github.com", "code is available", "open source", "open-source",
    "source code", "publicly available", "implementation is released",
    "our code", "code repository", "code will be released"
]

# ========== 筛选与评分配置 ==========
MIN_SCORE = 60          # 及格分数线
MAX_PAPERS = 8          # 每日最多推送数
OPEN_SOURCE_BONUS = 5   # 开源论文额外加分
ONLY_PUSH_OPENSOURCE = False  # 只推送开源论文开关（True/False）
ENABLE_PAPERS_WITH_CODE = True # 启用Papers with Code权威校验

# arXiv分区
ARXIV_CATEGORIES = ["cs.RO", "cs.CV", "cs.AI"]
RECORD_FILE = "pushed_papers.json"

# 环境变量
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
FEISHU_WEBHOOK = os.getenv("FEISHU_WEBHOOK", "")

# ===================== 工具函数 =====================
def load_pushed_records():
    if os.path.exists(RECORD_FILE):
        with open(RECORD_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_pushed_records(pushed_set):
    with open(RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump(list(pushed_set), f, ensure_ascii=False, indent=2)

def rule_based_score(title, abstract):
    """第一层：规则打分，返回基础分+是否通过初筛"""
    text = (title + " " + abstract).lower()
    base_score = 0
    
    # 关键词命中加分
    for kw, weight in DOMAIN_KEYWORDS.items():
        if kw.lower() in text:
            base_score += weight * 2
    
    # 优质机构/顶会加分
    for kw in BONUS_KEYWORDS:
        if kw.lower() in text:
            base_score += 5
    
    # 减分项：纯综述、无实验
    if re.search(r"review|survey|overview", text) and "we propose" not in text:
        base_score -= 20
    
    return base_score, base_score >= 10

def check_opensource_local(title, abstract):
    """本地关键词快筛，判断是否提及开源"""
    text = (title + " " + abstract).lower()
    for kw in CODE_KEYWORDS:
        if kw.lower() in text:
            # 尝试提取github链接
            url_match = re.search(r"https?://github\.com/[\w\-/]+", text)
            if url_match:
                return True, url_match.group(0)
            return True, None
    return False, None

def check_opensource_pwc(arxiv_id):
    """调用Papers with Code API，权威查询是否有开源代码"""
    if not ENABLE_PAPERS_WITH_CODE:
        return False, None
    try:
        # 清理arxiv ID（去掉版本号v1/v2）
        clean_id = re.sub(r"v\d+$", "", arxiv_id)
        url = f"https://paperswithcode.com/api/v1/papers/?arxiv_id={clean_id}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("count", 0) > 0:
            paper = data["results"][0]
            repos = paper.get("repositories", [])
            if repos:
                # 取官方/星数最高的仓库
                official = [r for r in repos if r.get("is_official")]
                target = official[0] if official else repos[0]
                return True, target.get("url", "")
        return False, None
    except Exception as e:
        print(f"Papers with Code查询失败:{e}")
        return False, None

def get_opensource_info(title, abstract, arxiv_id):
    """整合双层识别，返回代码状态描述和链接"""
    # 先本地快筛
    local_open, local_url = check_opensource_local(title, abstract)
    # 再权威校验
    pwc_open, pwc_url = check_opensource_pwc(arxiv_id)
    
    if pwc_open:
        status = "✅ 已开源"
        link = pwc_url
    elif local_open:
        status = "🟡 摘要提及开源"
        link = local_url if local_url else "未提取到明确链接"
    else:
        status = "❌ 未发现开源代码"
        link = ""
    
    is_open = pwc_open or local_open
    return is_open, status, link

def llm_quality_score(title, abstract):
    """第二层：大模型质量打分，返回0-100分"""
    if not ZHIPU_API_KEY:
        return 70
    
    prompt = f"""你是SLAM与激光雷达领域的资深算法专家，请对以下论文进行质量评分，总分0-100分。
评分维度：
1. 领域相关性（30分）：与激光SLAM、点云处理、机器人感知、自动驾驶激光感知的贴合度
2. 技术创新性（30分）：是否提出了新的算法、框架或解决方案，而非简单改进
3. 实验完整性（25分）：是否有公开数据集验证、对比实验、消融实验
4. 工程价值（15分）：是否具备落地可行性、对工业界有参考意义

论文标题：{title}
论文摘要：{abstract}

输出要求：只输出一个整数分数，不要任何解释、多余文字和单位。"""
    
    headers = {"Authorization": f"Bearer {ZHIPU_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "glm-4-flash",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    }
    try:
        res = requests.post("https://open.bigmodel.cn/api/paas/v4/chat/completions",
                            json=payload, headers=headers, timeout=30)
        score_text = res.json()["choices"][0]["message"]["content"].strip()
        score = int(re.findall(r"\d+", score_text)[0])
        return min(max(score, 0), 100)
    except Exception as e:
        print(f"质量打分失败:{e}")
        return 60

def llm_structured_summary(title, abstract):
    """生成结构化技术概述"""
    if not ZHIPU_API_KEY:
        return f"【原文摘要】{abstract[:400]}..."
    
    prompt = f"""你是激光SLAM与点云处理领域专家，请用中文对论文进行结构化总结，严格按以下5个小标题输出，每个标题1-2句话，简洁专业：
1. 核心痛点
2. 技术方案
3. 关键创新
4. 实验表现
5. 适用方向

论文标题：{title}
论文摘要：{abstract}

输出要求：不要多余开场白，直接按5个小标题输出，不要使用markdown格式，纯文本分段。"""
    
    headers = {"Authorization": f"Bearer {ZHIPU_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "glm-4-flash",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    try:
        res = requests.post("https://open.bigmodel.cn/api/paas/v4/chat/completions",
                            json=payload, headers=headers, timeout=60)
        return res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"概述生成失败:{e}")
        return f"【原文摘要】{abstract[:400]}..."

def fetch_arxiv_papers():
    """抓取arXiv最新论文"""
    base_url = "http://export.arxiv.org/api/query"
    query_str = " OR ".join([f"all:{kw}" for kw in list(DOMAIN_KEYWORDS.keys())[:10]])
    cat_str = " OR ".join([f"cat:{cat}" for cat in ARXIV_CATEGORIES])
    full_query = f"({query_str}) AND ({cat_str})"
    params = {
        "search_query": full_query,
        "start": 0,
        "max_results": 60,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    resp = requests.get(base_url, params=params, timeout=30)
    import xml.etree.ElementTree as ET
    root = ET.fromstring(resp.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("atom:entry", ns):
        paper_id = entry.find("atom:id", ns).text.split("/abs/")[-1]
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        abstract = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
        pdf_url = entry.find("atom:link[@type='application/pdf']", ns).attrib["href"]
        authors = [auth.find("atom:name", ns).text for auth in entry.findall("atom:author", ns)]
        submit_date = entry.find("atom:updated", ns).text.split("T")[0]
        papers.append({
            "id": paper_id, "title": title, "abstract": abstract,
            "pdf": pdf_url, "authors": authors, "date": submit_date
        })
    return papers

def send_feishu_msg(paper_list):
    """飞书推送结构化日报"""
    if not FEISHU_WEBHOOK or len(paper_list) == 0:
        print("无推送渠道或无优质论文")
        return
    today = datetime.now().strftime("%Y-%m-%d")
    content = f"📡 SLAM&激光雷达每日优质论文日报 {today}\n今日共筛选出 {len(paper_list)} 篇高分论文\n\n"
    
    for idx, p in enumerate(paper_list, 1):
        summary = llm_structured_summary(p["title"], p["abstract"])
        content += f"━━━ 第{idx}篇  综合评分：{p['final_score']}分 ━━━\n"
        content += f"标题：{p['title']}\n"
        content += f"作者：{','.join(p['authors'][:3])}{'等' if len(p['authors'])>3 else ''}\n"
        content += f"代码状态：{p['code_status']}"
        if p["code_link"]:
            content += f"  {p['code_link']}"
        content += "\n"
        content += f"PDF链接：{p['pdf']}\n"
        content += f"--- 技术概述 ---\n{summary}\n\n"
        time.sleep(1)
    
    send_data = {"msg_type": "text", "content": {"text": content}}
    requests.post(FEISHU_WEBHOOK, json=send_data)
    print("推送完成")

# ===================== 主流程 =====================
if __name__ == "__main__":
    pushed_ids = load_pushed_records()
    all_papers = fetch_arxiv_papers()
    
    # 1. 去重
    new_papers = [p for p in all_papers if p["id"] not in pushed_ids]
    print(f"抓取到今日新论文 {len(new_papers)} 篇")
    
    # 2. 规则初筛
    candidate_papers = []
    for p in new_papers:
        base_score, pass_rule = rule_based_score(p["title"], p["abstract"])
        if pass_rule:
            p["base_score"] = base_score
            candidate_papers.append(p)
    print(f"规则初筛通过 {len(candidate_papers)} 篇")
    
    # 3. 开源识别 + 质量打分 + 综合评分
    scored_papers = []
    for p in candidate_papers:
        # 代码开源识别
        is_open, code_status, code_link = get_opensource_info(p["title"], p["abstract"], p["id"])
        p["is_open"] = is_open
        p["code_status"] = code_status
        p["code_link"] = code_link
        
        # 只推送开源论文开关
        if ONLY_PUSH_OPENSOURCE and not is_open:
            continue
        
        # 质量打分
        llm_score = llm_quality_score(p["title"], p["abstract"])
        
        # 综合分：规则分40% + LLM分60% + 开源加分
        final_score = int(p["base_score"] * 0.4 + llm_score * 0.6)
        if is_open:
            final_score += OPEN_SOURCE_BONUS
        p["final_score"] = min(final_score, 100)
        
        if p["final_score"] >= MIN_SCORE:
            scored_papers.append(p)
        
        time.sleep(0.8)  # 控制请求频率，避免限流
    
    # 按分数降序排序，取Top N
    scored_papers.sort(key=lambda x: x["final_score"], reverse=True)
    top_papers = scored_papers[:MAX_PAPERS]
    print(f"优质论文筛选完成，共 {len(top_papers)} 篇")
    
    # 4. 推送
    if top_papers:
        send_feishu_msg(top_papers)
        new_ids = set(p["id"] for p in top_papers)
        pushed_ids.update(new_ids)
        save_pushed_records(pushed_ids)
        print(f"已推送 {len(top_papers)} 篇，记录已更新")
    else:
        print("今日无符合标准的优质论文")
