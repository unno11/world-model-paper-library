"""Generate public paper navigation from catalog.csv; preserve article bodies."""
from pathlib import Path
from urllib.parse import quote,unquote
import csv,re,os
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[1]
def relative(target,current):return quote(os.path.relpath(target,current.parent).replace('\\','/'),safe='/#')
def clean(s):return str(s).replace('|','\\|').replace('\n',' ')
def write_table(path,title,intro,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    lines=[f'# {title}','',intro,'','| 编号 | 论文 | 刊会 | 年份 |','| --- | --- | --- | --- |']
    for r in sorted(rows,key=lambda x:(x['topic'],x['title'].lower())):
        lines.append(f'| {r["arxiv_id"]} | [{clean(r["title"])}]({relative(ROOT/r["path"],path)}) | {clean(r["venue"])} | {r["year"] or "未注明"} |')
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
def refresh():
    with (ROOT/'catalog.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    ids=set();paths=set();groups=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if r['arxiv_id'] in ids:raise ValueError('Duplicate paper ID: '+r['arxiv_id'])
        ids.add(r['arxiv_id']);p=(ROOT/r['path']).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file():raise ValueError('Missing or unsafe path: '+r['path'])
        if p in paths:raise ValueError('Duplicate article path')
        paths.add(p)
        if p.relative_to(ROOT).parts[0]!=r['topic']:raise ValueError('Topic and path disagree')
        groups[r['topic']][p.parent.name].append(r)
    write_table(ROOT/'00_导航/全部论文.md','全部论文目录','按编号、题名、刊会检索。每个标题链接到唯一论文页，正文包含官方资料或论文解析。',rows)
    venues=defaultdict(list)
    for r in rows:venues[r['venue']].append(r)
    vpage=ROOT/'00_导航/按刊会查找.md'
    vl=['# 按刊会查找','','保留主目录“研究主题 → 刊会 → 论文”；这里提供跨主题的刊会入口。','']
    for venue,items in sorted(venues.items()):
        name=re.sub(r'[/\\:*?"<>|]','-',venue)
        vp=ROOT/'00_导航/刊会'/f'{name}.md'
        vl.append(f'- [{venue}]({relative(vp,vpage)})：{len(items)} 篇')
        write_table(vp,venue,'该刊会在不同研究主题下的论文。来源与发表信息核对范围见各论文页。',items)
    vpage.write_text('\n'.join(vl)+'\n',encoding='utf-8')
    home=['# 世界模型论文与解析','','收录世界模型及其相关表示学习、规划、控制与评测研究，按研究主题和刊会组织，提供论文资料与中文解析。','',
    '## 论文导航','','- [全部论文目录](00_%E5%AF%BC%E8%88%AA/%E5%85%A8%E9%83%A8%E8%AE%BA%E6%96%87.md)','- [按刊会查找](00_%E5%AF%BC%E8%88%AA/%E6%8C%89%E5%88%8A%E4%BC%9A%E6%9F%A5%E6%89%BE.md)','- [基础知识与方法谱系](基础知识与方法谱系/README.md)','- [引用与资料核对说明](SOURCES.md)','- [内容维护指南](CONTRIBUTING.md)','',
    '## 按研究主题查找','','| 研究主题 | 论文数 |','| --- | --- |']
    for topic,vs in sorted(groups.items()):
        items=[r for rs in vs.values() for r in rs]
        tp=ROOT/topic/'README.md'
        home.append(f'| [{topic}]({relative(tp,ROOT/"README.md")}) | {len(items)} |')
        tl=[f'# {topic}','','[返回首页](../README.md)','','| 刊会 | 论文数 |','| --- | --- |']
        for v,rs in sorted(vs.items()):
            vp=ROOT/topic/v/'README.md'
            tl.append(f'| [{v}]({relative(vp,tp)}) | {len(rs)} |')
            write_table(vp,f'{topic} / {v}','论文题名链接至资料与解析，原文和版本信息见论文页。',rs)
        tp.write_text('\n'.join(tl)+'\n',encoding='utf-8')
    home+=['','## 关于本库','',f'当前收录 {len(rows)} 篇论文，涵盖世界模型、表示学习、规划与控制、机器人操控及相关应用。可以从研究主题入手，也可以按刊会浏览或通过论文目录查找。','',
    '论文条目提供原文入口、作者、版本与发表信息；中文解析围绕研究问题、方法原理、关键实验与适用局限展开，帮助读者理解方法并联系相关工作。','',
    '欢迎通过 Issues 补充相关论文、讨论方法或指出资料错误。引用与版本信息见 [资料来源说明](SOURCES.md)，内容贡献方式见 [贡献指南](CONTRIBUTING.md)。']
    (ROOT/'README.md').write_text('\n'.join(home)+'\n',encoding='utf-8')
    basic=ROOT/'基础知识与方法谱系/README.md'
    basic.write_text('\n'.join(['# 基础知识与方法谱系','',*[f'- [{p.stem}]({relative(p,basic)})' for p in sorted(basic.parent.rglob('*.md')) if p!=basic],'']),encoding='utf-8')
    broken=[]
    for page in ROOT.rglob('*.md'):
        for t in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',page.read_text(encoding='utf-8')):
            if not re.match(r'https?://|mailto:|#|data:',t) and not (page.parent/unquote(t.split('#')[0])).is_file():broken.append((str(page.relative_to(ROOT)),t))
    if broken:raise ValueError('Broken local links: '+repr(broken[:20]))
    print(f'Public index: {len(rows)} papers; local links passed.')
if __name__=='__main__':refresh()
