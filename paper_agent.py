import requests
import json
import time
import os
import re
import hmac
import hashlib
import base64
import urllib.parse
import smtplib
import feedparser
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr
from datetime import datetime

# ========== 可配置参数区 ==========
# 分层关键词：核心强相关 / 泛垂直 / 泛领域，适配论文+中文文章双场景
DOMAIN_KEYWORDS = {
    # ========== 核心强相关（高权重，命中即强匹配）==========
    # 英文核心
    "lidar": 4, "point cloud": 4, "slam": 4, "lio": 4, "loam": 4,
    "point cloud registration": 5, "loop closure": 5, "imu fusion": 4,
    "odometry": 4, "gaussian splatting slam": 5, "nerf slam": 5,
    # 中文核心
    "激光雷达": 4, "点云": 4, "激光SLAM": 5, "点云配准": 5,
    "回环检测": 5, "惯导融合": 4, "里程计": 4, "点云分割": 4,
    "固态激光雷达": 4, "多线雷达": 4,

    # ========== 泛垂直相关（中权重，技术方向沾边）==========
    # 英文
    "livox": 3, "semantic segmentation": 3, "localization": 3,
    "feature extraction": 3, "voxel": 3,
    # 中文
    "览沃": 3, "体素": 3, "特征提取": 3,
    "自动驾驶感知": 3, "机器人定位": 3, "3D点云": 3,
    "多传感器融合": 3, "紧耦合": 3, "3D视觉": 3,
    "机器人导航": 3,

    # ========== 泛领域补充（低权重，仅作辅助）==========
    "autonomous driving": 2, "机器人": 2, "自动驾驶": 2, "人工智能": 1
}

# 优质机构/顶会关键词（命中额外加分）
BONUS_KEYWORDS = [
    "ICRA", "IROS", "CVPR", "ECCV", "3DV", "RSS", "TRO",
    "CMU", "MIT", "Stanford", "Oxford", "HKUST", "Tsinghua",
    "Waymo", "Tesla", "Huawei", "Baidu", "DJI", "RoboSense", "Hesai"
]

# 代码开源识别关键词
CODE_KEYWORDS = [
    "github.com", "code is available", "open source", "open-source",
    "source code", "publicly available", "implementation is released",
    "our code", "code repository", "code will be released", "开源"
]

# ========== 中文RSS源配置（海外节点稳定可访问版）==========
CN_RSS_SOURCES = [
    # 综合科技媒体（行业资讯+技术解读）
    {"name": "量子位", "url": "https://www.qbitai.com/feed"},
    {"name": "新智元", "url": "https://www.aiera.com/feed"},
    {"name": "开源中国-AI", "url": "https://www.oschina.net/news/rss?catalog=2"},
    # 技术社区（偏技术实践，匹配度更高）
    {"name": "掘金-人工智能", "url": "https://juejin.cn/rss/category/ai"},
    {"name": "思否-人工智能", "url": "https://segmentfault.com/feeds/channel/ai"},
]
MAX_CN_ARTICLES = 5  # 每日最多推送中文文章数

# ========== 筛选评分配置 ==========
MIN_SCORE = 60                  # 论文及格分数线
MIN_CN_SCORE = 45               # 中文文章及格分数线
MAX_PAPERS = 8                  # 每日最多推送论文数量
OPEN_SOURCE_BONUS = 5           # 开源论文额外加分
ONLY_PUSH_OPENSOURCE = False    # 只推送开源论文开关
ENABLE_PAPERS_WITH_CODE = True  # 启用Papers with Code权威开源校验

# 打分归一化基准值（规则分满分参考）
PAPER_RULE_MAX = 80     # 论文规则分理论满分
CN_RULE_MAX = 60        # 中文文章规则分理论满分

# arXiv分区
ARXIV_CATEGORIES = ["cs.RO", "cs.CV", "cs.AI"]
RECORD_FILE = "pushed_papers.json"
CN_RECORD_FILE = "pushed_cn_articles.json"

# ========== 环境变量（密钥从GitHub Secrets读取）==========
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
FEISHU_WEBHOOK = os.getenv("FEISHU_WEBHOOK", "")
DINGTALK_WEBHOOK = os.getenv("DINGTALK_WEBHOOK", "")
DINGTALK_SECRET = os.getenv("DINGTALK_SECRET", "")
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "")
EMAIL_SMTP_PASSWORD = os.getenv("EMAIL_SMTP_PASSWORD", "")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "")
SMTP_HOST = "smtp.qq.com"
SMTP_PORT = 465

# ===================== 基础工具函数 =====================
def load_json_set(filename):
    """加载JSON格式的去重记录"""
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_json_set(filename, data_set):
    """保存JSON格式的去重记录"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(list(data_set), f, ensure_ascii=False, indent=2)

# ===================== 论文抓取（arXiv官方API） =====================
def fetch_arxiv_papers():
    """调用arXiv官方API抓取最新论文"""
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
            "pdf": pdf_url, "authors": authors, "date": submit_date, "type": "paper"
        })
    return papers

# ===================== 中文RSS文章抓取 =====================
def fetch_cn_rss_articles():
    """抓取所有中文RSS源，增加反爬头、状态校验、错误详情打印"""
    all_articles = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept": "application/rss+xml, application/xml, text/xml;q=0.9"
    }
    
    for source in CN_RSS_SOURCES:
        print(f"正在抓取: {source['name']} - {source['url']}")
        try:
            resp = requests.get(source["url"], headers=headers, timeout=20)
            if resp.status_code != 200:
                print(f"  ❌ {source['name']} 请求失败，状态码: {resp.status_code}")
                continue
            
            feed = feedparser.parse(resp.content)
            if feed.bozo != 0:
                print(f"  ⚠️  {source['name']} 解析警告: {feed.bozo_exception}")
            
            entry_count = len(feed.entries)
            print(f"  ✅ 成功抓取 {entry_count} 篇文章")
            
            for entry in feed.entries[:15]:
                article_id = entry.link.strip()
                title = entry.title.strip()
                summary = entry.get("summary", entry.get("description", "")).strip()
                summary = re.sub(r"<[^>]+>", "", summary)
                summary = re.sub(r"\s+", " ", summary)
                
                all_articles.append({
                    "id": article_id,
                    "title": title,
                    "abstract": summary,
                    "link": article_id,
                    "source": source["name"],
                    "date": entry.get("published", ""),
                    "type": "cn_article"
                })
            time.sleep(1)
        except Exception as e:
            print(f"  ❌ {source['name']} 抓取异常: {str(e)}")
    return all_articles

# ===================== 规则打分筛选 =====================
def rule_based_score(title, abstract):
    """英文论文规则初筛打分"""
    text = (title + " " + abstract).lower()
    base_score = 0
    
    for kw, weight in DOMAIN_KEYWORDS.items():
        if kw.lower() in text:
            base_score += weight * 2
    
    for kw in BONUS_KEYWORDS:
        if kw.lower() in text:
            base_score += 5
    
    # 纯综述无创新工作减分
    if re.search(r"review|survey|overview", text) and "we propose" not in text:
        base_score -= 20
    
    return base_score, base_score >= 10

def rule_score_cn_article(title, abstract):
    """中文文章规则打分，标题命中权重翻倍，门槛下调至3分，宽进严出"""
    title_lower = title.lower()
    text_lower = (title + " " + abstract).lower()
    score = 0
    
    for kw, weight in DOMAIN_KEYWORDS.items():
        kw_lower = kw.lower()
        if kw_lower in title_lower:
            score += weight * 2
        elif kw_lower in text_lower:
            score += weight
    
    # 调试日志：打印得分≥3分的文章，方便调参
    if score >= 3:
        print(f"  [规则得分{score}] {title[:45]}...")
    
    # 初筛门槛降至3分，宽进严出，后续大模型二次筛选
    return score, score >= 3

# ===================== 开源代码识别 =====================
def check_opensource_local(title, abstract):
    """本地关键词快筛，提取摘要中的开源信息"""
    text = (title + " " + abstract).lower()
    for kw in CODE_KEYWORDS:
        if kw.lower() in text:
            url_match = re.search(r"https?://github\.com/[\w\-/]+", text)
            if url_match:
                return True, url_match.group(0)
            return True, None
    return False, None

def check_opensource_pwc(arxiv_id):
    """调用Papers with Code官方API，增加防拦截与异常校验"""
    if not ENABLE_PAPERS_WITH_CODE:
        return False, None
    try:
        clean_id = re.sub(r"v\d+$", "", arxiv_id)
        url = f"https://paperswithcode.com/api/v1/papers/?arxiv_id={clean_id}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        
        if resp.status_code != 200:
            return False, None
        
        try:
            data = resp.json()
        except ValueError:
            return False, None
            
        if data.get("count", 0) > 0:
            paper = data["results"][0]
            repos = paper.get("repositories", [])
            if repos:
                official = [r for r in repos if r.get("is_official")]
                target = official[0] if official else repos[0]
                return True, target.get("url", "")
        return False, None
    except Exception:
        return False, None

def get_opensource_info(title, abstract, arxiv_id):
    """整合双层识别，返回最终开源状态"""
    local_open, local_url = check_opensource_local(title, abstract)
    pwc_open, pwc_url = check_opensource_pwc(arxiv_id)
    
    if pwc_open:
        return True, "✅ 已开源", pwc_url
    elif local_open:
        return True, "🟡 提及开源", local_url if local_url else "无明确链接"
    else:
        return False, "❌ 未发现开源", ""

# ===================== 大模型能力 =====================
def llm_quality_score(title, abstract, is_cn=False):
    """大模型质量打分，支持论文/中文文章两种模式"""
    if not ZHIPU_API_KEY:
        return 70
    
    if is_cn:
        prompt = f"""请对这篇中文技术文章进行质量评分，总分0-100分。
评分维度：
1. 领域相关性（40分）：与激光SLAM、点云处理、自动驾驶感知、机器人技术的贴合度
2. 技术深度（30分）：是否有具体技术细节、算法讲解，而非泛泛资讯
3. 信息价值（30分）：对从业者的参考价值

标题：{title}
摘要：{abstract}

输出要求：只输出一个整数分数，不要任何解释和多余文字。"""
    else:
        prompt = f"""你是SLAM与激光雷达领域资深算法专家，对论文进行0-100分综合评分。
评分维度：
1. 领域相关性（30分）：与激光SLAM、点云处理、机器人感知的贴合度
2. 技术创新性（30分）：是否提出新算法/新框架，而非简单堆叠改进
3. 实验完整性（25分）：是否有公开数据集、对比实验、消融实验
4. 工程价值（15分）：是否具备落地可行性、工业参考价值

论文标题：{title}
论文摘要：{abstract}

输出要求：只输出一个整数分数，不要任何解释和多余文字。"""
    
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
    except Exception:
        return 60

def llm_structured_summary(title, abstract, is_cn=False):
    """生成结构化中文技术概述"""
    if not ZHIPU_API_KEY:
        return abstract[:300] + "..."
    
    if is_cn:
        prompt = f"""用3句话总结这篇文章的核心内容，突出技术亮点和行业价值，简洁专业：
标题：{title}
摘要：{abstract}
不要多余开场白，直接输出总结内容。"""
    else:
        prompt = f"""你是激光SLAM与点云处理领域专家，用中文对论文做结构化总结，严格按以下5个小标题输出，每个标题1-2句话，简洁专业：
1. 核心痛点
2. 技术方案
3. 关键创新
4. 实验表现
5. 适用方向

论文标题：{title}
论文摘要：{abstract}

输出要求：不要开场白，直接按5个小标题输出，纯文本分段。"""
    
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
    except Exception:
        return abstract[:300] + "..."

# ===================== 推送渠道 =====================
def build_full_content(paper_list, cn_list):
    """拼接完整推送内容，分两大板块"""
    today = datetime.now().strftime("%Y-%m-%d")
    content = f"📡 SLAM&激光雷达每日资讯日报 {today}\n"
    content += f"📄 顶会论文 {len(paper_list)} 篇 ｜ 📰 中文技术精选 {len(cn_list)} 篇\n\n"

    if paper_list:
        content += "═══════ 【一、国际顶会论文精选】 ═══════\n\n"
        for idx, p in enumerate(paper_list, 1):
            summary = llm_structured_summary(p["title"], p["abstract"])
            content += f"━━━ 第{idx}篇  评分：{p['final_score']}分 ━━━\n"
            content += f"标题：{p['title']}\n"
            content += f"作者：{','.join(p['authors'][:3])}等\n"
            content += f"代码：{p['code_status']}"
            if p["code_link"]:
                content += f" {p['code_link']}"
            content += f"\nPDF：{p['pdf']}\n"
            content += f"技术概述：{summary}\n\n"
            time.sleep(0.8)

    if cn_list:
        content += "═══════ 【二、国内技术文章精选】 ═══════\n\n"
        for idx, a in enumerate(cn_list, 1):
            summary = llm_structured_summary(a["title"], a["abstract"], is_cn=True)
            content += f"━━━ 第{idx}篇 [{a['source']}]  评分：{a['final_score']}分 ━━━\n"
            content += f"标题：{a['title']}\n"
            content += f"链接：{a['link']}\n"
            content += f"内容摘要：{summary}\n\n"
            time.sleep(0.8)
    
    return content

def send_feishu_msg(content):
    """飞书推送"""
    if not FEISHU_WEBHOOK:
        print("飞书未配置，跳过推送")
        return
    try:
        requests.post(FEISHU_WEBHOOK, json={"msg_type": "text", "content": {"text": content}}, timeout=15)
        print("飞书推送成功")
    except Exception as e:
        print(f"飞书推送失败: {e}")

def dingtalk_sign():
    """钉钉加签生成"""
    if not DINGTALK_SECRET:
        return {}
    timestamp = str(round(time.time() * 1000))
    secret_enc = DINGTALK_SECRET.encode("utf-8")
    string_to_sign = "{}\n{}".format(timestamp, DINGTALK_SECRET)
    hmac_code = hmac.new(secret_enc, string_to_sign.encode("utf-8"), digestmod=hashlib.sha256).digest()
    sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
    return {"timestamp": timestamp, "sign": sign}

def send_dingtalk_msg(content):
    """钉钉推送"""
    if not DINGTALK_WEBHOOK:
        print("钉钉未配置，跳过推送")
        return
    try:
        sign_params = dingtalk_sign()
        url = DINGTALK_WEBHOOK
        if sign_params:
            url += f"&timestamp={sign_params['timestamp']}&sign={sign_params['sign']}"
        resp = requests.post(url, json={"msgtype": "text", "text": {"content": content}}, timeout=15)
        result = resp.json()
        if result.get("errcode") == 0:
            print("钉钉推送成功")
        else:
            print(f"钉钉推送失败: {result.get('errmsg')}")
    except Exception as e:
        print(f"钉钉推送异常: {e}")

def send_email(content):
    """邮箱推送（修复From头格式，严格符合RFC规范）"""
    if not EMAIL_SENDER or not EMAIL_SMTP_PASSWORD or not EMAIL_RECEIVER:
        print("邮箱未配置，跳过推送")
        return
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        msg = MIMEText(content, "plain", "utf-8")
        msg["From"] = formataddr(("论文日报Agent", EMAIL_SENDER), "utf-8")
        msg["To"] = formataddr(("收件人", EMAIL_RECEIVER), "utf-8")
        msg["Subject"] = Header(f"【每日资讯】SLAM&激光雷达日报 {today}", "utf-8")

        server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT)
        server.login(EMAIL_SENDER, EMAIL_SMTP_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        print("邮件推送成功")
    except Exception as e:
        print(f"邮件推送失败: {e}")

# ===================== 主执行流程 =====================
if __name__ == "__main__":
    # ========== 第一部分：处理国际论文 ==========
    pushed_paper_ids = load_json_set(RECORD_FILE)
    all_papers = fetch_arxiv_papers()
    new_papers = [p for p in all_papers if p["id"] not in pushed_paper_ids]
    print(f"\n抓取到今日新论文 {len(new_papers)} 篇")

    candidate_papers = []
    for p in new_papers:
        base_score, pass_rule = rule_based_score(p["title"], p["abstract"])
        if pass_rule:
            p["base_score"] = base_score
            candidate_papers.append(p)

    scored_papers = []
    for p in candidate_papers:
        is_open, code_status, code_link = get_opensource_info(p["title"], p["abstract"], p["id"])
        p["is_open"] = is_open
        p["code_status"] = code_status
        p["code_link"] = code_link
        
        if ONLY_PUSH_OPENSOURCE and not is_open:
            continue
        
        llm_score = llm_quality_score(p["title"], p["abstract"])
        # 修复：规则分归一化到100分制，再加权计算综合分
        normalized_rule = min(p["base_score"] / PAPER_RULE_MAX * 100, 100)
        final_score = int(normalized_rule * 0.4 + llm_score * 0.6)
        
        if is_open:
            final_score += OPEN_SOURCE_BONUS
        p["final_score"] = min(final_score, 100)
        
        if p["final_score"] >= MIN_SCORE:
            scored_papers.append(p)
        
        time.sleep(1)

    scored_papers.sort(key=lambda x: x["final_score"], reverse=True)
    top_papers = scored_papers[:MAX_PAPERS]
    print(f"筛选出优质论文 {len(top_papers)} 篇")

    # ========== 第二部分：处理中文文章（全局异常兜底）==========
    top_cn = []
    try:
        pushed_cn_ids = load_json_set(CN_RECORD_FILE)
        all_cn = fetch_cn_rss_articles()
        new_cn = [a for a in all_cn if a["id"] not in pushed_cn_ids]
        print(f"\n抓取到中文新文章 {len(new_cn)} 篇")

        candidate_cn = []
        for a in new_cn:
            base_score, pass_rule = rule_score_cn_article(a["title"], a["abstract"])
            if pass_rule:
                a["base_score"] = base_score
                candidate_cn.append(a)

        scored_cn = []
        for a in candidate_cn:
            try:
                llm_score = llm_quality_score(a["title"], a["abstract"], is_cn=True)
                # 修复：规则分归一化，泛源场景大模型占80%权重
                normalized_rule = min(a["base_score"] / CN_RULE_MAX * 100, 100)
                final_score = int(normalized_rule * 0.2 + llm_score * 0.8)
                a["final_score"] = min(final_score, 100)
                if a["final_score"] >= MIN_CN_SCORE:
                    scored_cn.append(a)
            except Exception as e:
                print(f"单篇中文文章处理异常，跳过: {e}")
            time.sleep(0.8)

        scored_cn.sort(key=lambda x: x["final_score"], reverse=True)
        top_cn = scored_cn[:MAX_CN_ARTICLES]
        print(f"筛选出优质中文文章 {len(top_cn)} 篇")
    except Exception as e:
        print(f"\n⚠️  中文文章模块整体异常，已跳过，不影响论文推送: {e}")

    # ========== 第三部分：多渠道统一推送 ==========
    if top_papers or top_cn:
        full_content = build_full_content(top_papers, top_cn)
        print("\n===== 开始推送 =====")
        send_feishu_msg(full_content)
        send_dingtalk_msg(full_content)
        send_email(full_content)

        if top_papers:
            new_paper_ids = set(p["id"] for p in top_papers)
            pushed_paper_ids.update(new_paper_ids)
            save_json_set(RECORD_FILE, pushed_paper_ids)
        
        if top_cn:
            new_cn_ids = set(a["id"] for a in top_cn)
            pushed_cn_ids = load_json_set(CN_RECORD_FILE)
            pushed_cn_ids.update(new_cn_ids)
            save_json_set(CN_RECORD_FILE, pushed_cn_ids)
        print("\n所有记录已更新，执行完成")
    else:
        print("\n今日无符合条件的内容")
