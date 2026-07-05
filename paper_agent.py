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
# 核心领域关键词（权重越高，命中后相关性加分越多）
DOMAIN_KEYWORDS = {
    # ========== 核心高权重（命中即强相关）==========
    # 英文核心
    "lidar": 4, "point cloud": 4, "slam": 4, "lio": 4, "loam": 4,
    "point cloud registration": 5, "loop closure": 5, "imu fusion": 4,
    "odometry": 4,
    # 中文核心
    "激光雷达": 4, "点云": 4, "建图": 3, "激光SLAM": 5, "点云配准": 5,
    "回环检测": 5, "惯导融合": 4, "里程计": 4, "点云分割": 4,

    # ========== 次高权重（细分方向）==========
    # 英文细分
    "livox": 3, "gaussian splatting slam": 5, "nerf slam": 5,
    "semantic segmentation": 3, "localization": 3, "feature extraction": 3,
    "voxel": 3, "autonomous driving": 2,
    # 中文细分
    "固态激光雷达": 4, "览沃": 3, "体素": 3, "特征提取": 3,
    "自动驾驶感知": 3, "机器人定位": 3, "3D点云": 4,
    "多传感器融合": 3, "紧耦合": 4, "松耦合": 3,

    # ========== 低权重（泛领域，仅作补充）==========
    "autonomous driving": 2, "机器人": 2, "自动驾驶": 2, "3D视觉": 2
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

# ========== 中文RSS源配置 ==========
# 新增：国内优质技术文章RSS源
CN_RSS_SOURCES = [
    {"name": "机器之心", "url": "https://www.jiqizhixin.com/rss"},
    {"name": "量子位", "url": "https://www.qbitai.com/feed"},
    {"name": "点云PCL", "url": "https://cloud.tencent.com/developer/column/rss/79230"},
    {"name": "自动驾驶之心", "url": "https://www.zdjszx.com/feed"},
]
MAX_CN_ARTICLES = 5  # 每日最多推送中文文章数

# ========== 筛选评分配置 ==========
MIN_SCORE = 60                  # 论文及格分数线
MIN_CN_SCORE = 50               # 中文文章及格分数线
MAX_PAPERS = 8                  # 每日最多推送论文数量
OPEN_SOURCE_BONUS = 5           # 开源论文额外加分
ONLY_PUSH_OPENSOURCE = False    # 只推送开源论文开关
ENABLE_PAPERS_WITH_CODE = True  # 启用Papers with Code权威开源校验

# arXiv分区
ARXIV_CATEGORIES = ["cs.RO", "cs.CV", "cs.AI"]
RECORD_FILE = "pushed_papers.json"
CN_RECORD_FILE = "pushed_cn_articles.json"

# ========== 环境变量 ==========
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
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_json_set(filename, data_set):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(list(data_set), f, ensure_ascii=False, indent=2)

# ===================== 论文抓取（原有arXiv） =====================
def fetch_arxiv_papers():
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

# ===================== 新增：中文RSS文章抓取 =====================
def fetch_cn_rss_articles():
    """抓取所有中文RSS源的最新文章"""
    all_articles = []
    for source in CN_RSS_SOURCES:
        try:
            resp = requests.get(source["url"], timeout=15)
            feed = feedparser.parse(resp.content)
            for entry in feed.entries[:10]:  # 每个源取最新10篇
                article_id = entry.link
                title = entry.title
                summary = entry.get("summary", "").strip()
                # 去除HTML标签
                summary = re.sub(r"<[^>]+>", "", summary)
                all_articles.append({
                    "id": article_id,
                    "title": title,
                    "abstract": summary,
                    "link": article_id,
                    "source": source["name"],
                    "date": entry.get("published", ""),
                    "type": "cn_article"
                })
            time.sleep(0.5)
        except Exception as e:
            print(f"抓取{source['name']}失败: {e}")
    return all_articles

def rule_score_cn_article(title, abstract):
    """中文文章规则打分，标题命中权重翻倍"""
    title_lower = title.lower()
    text_lower = (title + " " + abstract).lower()
    score = 0
    
    for kw, weight in DOMAIN_KEYWORDS.items():
        kw_lower = kw.lower()
        # 标题命中：权重翻倍
        if kw_lower in title_lower:
            score += weight * 2
        # 仅摘要命中：正常权重
        elif kw_lower in text_lower:
            score += weight
    
    return score, score >= 12  # 门槛同步提高，过滤弱相关文章

# ===================== 规则初筛 =====================
def rule_based_score(title, abstract):
    text = (title + " " + abstract).lower()
    base_score = 0
    for kw, weight in DOMAIN_KEYWORDS.items():
        if kw.lower() in text:
            base_score += weight * 2
    for kw in BONUS_KEYWORDS:
        if kw.lower() in text:
            base_score += 5
    if re.search(r"review|survey|overview", text) and "we propose" not in text:
        base_score -= 20
    return base_score, base_score >= 10

# ===================== 开源代码识别 =====================
def check_opensource_local(title, abstract):
    text = (title + " " + abstract).lower()
    for kw in CODE_KEYWORDS:
        if kw.lower() in text:
            url_match = re.search(r"https?://github\.com/[\w\-/]+", text)
            if url_match:
                return True, url_match.group(0)
            return True, None
    return False, None

def check_opensource_pwc(arxiv_id):
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
    except Exception as e:
        return False, None

def get_opensource_info(title, abstract, arxiv_id):
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
    if not ZHIPU_API_KEY:
        return 70
    if is_cn:
        prompt = f"""请对这篇中文技术文章进行质量评分，0-100分。维度：领域相关性40分、技术深度30分、信息价值30分。
标题：{title}
摘要：{abstract}
只输出一个整数分数，不要解释。"""
    else:
        prompt = f"""你是SLAM与激光雷达领域专家，对论文进行0-100分综合评分。
维度：相关性30、创新性30、实验完整性25、工程价值15。
标题：{title}
摘要：{abstract}
只输出一个整数分数，不要解释。"""
    
    headers = {"Authorization": f"Bearer {ZHIPU_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "glm-4-flash", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}
    try:
        res = requests.post("https://open.bigmodel.cn/api/paas/v4/chat/completions",
                            json=payload, headers=headers, timeout=30)
        score = int(re.findall(r"\d+", res.json()["choices"][0]["message"]["content"])[0])
        return min(max(score, 0), 100)
    except:
        return 60

def llm_structured_summary(title, abstract, is_cn=False):
    if not ZHIPU_API_KEY:
        return abstract[:300] + "..."
    if is_cn:
        prompt = f"""用3句话总结这篇文章的核心内容，突出技术亮点和行业价值，简洁专业：
标题：{title}
摘要：{abstract}"""
    else:
        prompt = f"""你是激光SLAM专家，用中文结构化总结论文，按5点输出：
1.核心痛点 2.技术方案 3.关键创新 4.实验表现 5.适用方向
标题：{title}
摘要：{abstract}
纯文本输出，不要markdown。"""
    
    headers = {"Authorization": f"Bearer {ZHIPU_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "glm-4-flash", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
    try:
        res = requests.post("https://open.bigmodel.cn/api/paas/v4/chat/completions",
                            json=payload, headers=headers, timeout=60)
        return res.json()["choices"][0]["message"]["content"].strip()
    except:
        return abstract[:300] + "..."

# ===================== 推送渠道 =====================
def build_full_content(paper_list, cn_list):
    """拼接完整推送内容，分两大板块"""
    today = datetime.now().strftime("%Y-%m-%d")
    content = f"📡 SLAM&激光雷达每日资讯日报 {today}\n"
    content += f"📄 顶会论文 {len(paper_list)} 篇 ｜ 📰 中文技术精选 {len(cn_list)} 篇\n\n"

    # 板块一：论文
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

    # 板块二：中文文章
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
    if not FEISHU_WEBHOOK:
        return
    requests.post(FEISHU_WEBHOOK, json={"msg_type": "text", "content": {"text": content}})
    print("飞书推送成功")

def dingtalk_sign():
    if not DINGTALK_SECRET:
        return {}
    timestamp = str(round(time.time() * 1000))
    secret_enc = DINGTALK_SECRET.encode("utf-8")
    string_to_sign = "{}\n{}".format(timestamp, DINGTALK_SECRET)
    hmac_code = hmac.new(secret_enc, string_to_sign.encode("utf-8"), digestmod=hashlib.sha256).digest()
    sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
    return {"timestamp": timestamp, "sign": sign}

def send_dingtalk_msg(content):
    if not DINGTALK_WEBHOOK:
        return
    sign_params = dingtalk_sign()
    url = DINGTALK_WEBHOOK
    if sign_params:
        url += f"&timestamp={sign_params['timestamp']}&sign={sign_params['sign']}"
    resp = requests.post(url, json={"msgtype": "text", "text": {"content": content}}, timeout=15)
    if resp.json().get("errcode") == 0:
        print("钉钉推送成功")
    else:
        print(f"钉钉推送失败: {resp.json().get('errmsg')}")

def send_email(content):
    if not EMAIL_SENDER or not EMAIL_SMTP_PASSWORD or not EMAIL_RECEIVER:
        return
    today = datetime.now().strftime("%Y-%m-%d")
    msg = MIMEText(content, "plain", "utf-8")
    msg["From"] = formataddr(("论文日报Agent", EMAIL_SENDER), "utf-8")
    msg["To"] = formataddr(("收件人", EMAIL_RECEIVER), "utf-8")
    msg["Subject"] = Header(f"【每日资讯】SLAM&激光雷达日报 {today}", "utf-8")
    try:
        server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT)
        server.login(EMAIL_SENDER, EMAIL_SMTP_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        print("邮件推送成功")
    except Exception as e:
        print(f"邮件推送失败: {e}")

# ===================== 主执行流程 =====================
if __name__ == "__main__":
    # ========== 处理论文部分 ==========
    pushed_paper_ids = load_json_set(RECORD_FILE)
    all_papers = fetch_arxiv_papers()
    new_papers = [p for p in all_papers if p["id"] not in pushed_paper_ids]
    print(f"抓取到新论文 {len(new_papers)} 篇")

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
        final_score = int(p["base_score"] * 0.4 + llm_score * 0.6)
        if is_open:
            final_score += OPEN_SOURCE_BONUS
        p["final_score"] = min(final_score, 100)
        if p["final_score"] >= MIN_SCORE:
            scored_papers.append(p)
        time.sleep(1)

    scored_papers.sort(key=lambda x: x["final_score"], reverse=True)
    top_papers = scored_papers[:MAX_PAPERS]

    # ========== 处理中文文章部分 ==========
    pushed_cn_ids = load_json_set(CN_RECORD_FILE)
    all_cn = fetch_cn_rss_articles()
    new_cn = [a for a in all_cn if a["id"] not in pushed_cn_ids]
    print(f"抓取到中文新文章 {len(new_cn)} 篇")

    candidate_cn = []
    for a in new_cn:
        base_score, pass_rule = rule_score_cn_article(a["title"], a["abstract"])
        if pass_rule:
            a["base_score"] = base_score
            candidate_cn.append(a)

    scored_cn = []
    for a in candidate_cn:
        llm_score = llm_quality_score(a["title"], a["abstract"], is_cn=True)
        final_score = int(a["base_score"] * 0.5 + llm_score * 0.5)
        a["final_score"] = min(final_score, 100)
        if a["final_score"] >= MIN_CN_SCORE:
            scored_cn.append(a)
        time.sleep(0.8)

    scored_cn.sort(key=lambda x: x["final_score"], reverse=True)
    top_cn = scored_cn[:MAX_CN_ARTICLES]
    print(f"筛选出优质中文文章 {len(top_cn)} 篇")

    # ========== 推送 ==========
    if top_papers or top_cn:
        full_content = build_full_content(top_papers, top_cn)
        send_feishu_msg(full_content)
        send_dingtalk_msg(full_content)
        send_email(full_content)

        # 更新记录
        new_paper_ids = set(p["id"] for p in top_papers)
        pushed_paper_ids.update(new_paper_ids)
        save_json_set(RECORD_FILE, pushed_paper_ids)

        new_cn_ids = set(a["id"] for a in top_cn)
        pushed_cn_ids.update(new_cn_ids)
        save_json_set(CN_RECORD_FILE, pushed_cn_ids)
        print("记录已更新")
    else:
        print("今日无符合条件的内容")
