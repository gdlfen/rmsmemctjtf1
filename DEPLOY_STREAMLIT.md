# 部署到 GitHub + Streamlit Community Cloud

目标：得到一个可以在手机浏览器打开的固定 URL，同时保留本地运行能力。

---

## 1. 准备 GitHub 仓库

在项目根目录执行（把 `<你的用户名>`、`<仓库名>` 替换掉）：

```bash
git init
git add .
git commit -m "feat: A股双轨投机雷达"
git branch -M main
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git push -u origin main
```

如果使用 SSH：

```bash
git remote add origin git@github.com:<你的用户名>/<仓库名>.git
git push -u origin main
```

> 仓库不要提交 `data/cache/`、`.venv/`、`.streamlit/secrets.toml`；`.gitignore` 已处理。
> `data/bootstrap/` 与 `data/validation.json` 建议提交，用于云端接口故障时兜底。

---

## 2. 创建 Streamlit Cloud App

1. 打开 <https://share.streamlit.io/>，用 GitHub 登录；
2. 点击 **Create app** / **New app**；
3. Repository 选择刚推送的仓库；
4. Branch 选 `main`；
5. Main file path 填：

```text
app.py
```

6. Advanced settings 中 Python version 选 **3.12**；
7. 不需要 Secrets、不需要数据库、不需要额外 packages.txt；
8. 点击 **Deploy**，等待安装依赖并启动。

首次加载会抓取全市场行情，可能需要 40–120 秒。之后会复用缓存。页面 URL 形如：

```text
https://<app-name>.streamlit.app
```

手机浏览器直接打开这个 URL 即可；建议添加到主屏幕。

---

## 3. 手机使用建议

- 首屏“总览”看两个榜单 Top 3；点“潜在投机榜 / 活跃投机榜”看完整表格；
- 表格可横向滑动；点“个股详情”查看雷达、价格/均线、风险标签；
- 页面顶部必须核对 **信号基准日** 与 **快照时间**；
- 出现“使用缓存/内置离线数据”时，说明实时接口异常，不要把它当作盘中实时信号；
- 侧栏“刷新行情（清空缓存）”会清空服务端缓存并重新抓取；
- Streamlit Cloud 免费实例在一段时间无访问后会休眠，下一次打开需要重新唤醒。

---

## 4. 云端接口不稳定怎么办

程序已内置三级降级：

1. 实时接口（东方财富 → 腾讯/新浪备用）；
2. 本机/实例磁盘缓存（最近一次成功数据）；
3. 仓库内置 `data/bootstrap/` 离线灾备包。

所有降级都会在页面顶部警告，且 `meta.stale=True`。

如果长期依赖云端，建议在每次部署前本地执行：

```bash
python scripts/build_bootstrap.py --depth 420
python scripts/run_validation.py
```

然后提交更新后的 `data/bootstrap/` 与 `data/validation.json`。这样即使云端完全无法访问行情接口，页面仍能展示“最后一次成功数据”，并明确标注基准日。

如果你使用的是 Streamlit Community Cloud，实例文件系统在重启后会重置；不要把“刷新缓存”当作永久数据存储。灾备包必须提交到 Git 才可靠。

---

## 5. 更新代码后

```bash
git add .
git commit -m "update: model / data"
git push
```

Streamlit Cloud 会自动检测到 push 并重新部署（取决于仓库设置）。部署完成后刷新手机页面。

---

## 6. 本地运行（与云端同一套代码）

```bash
pip install -r requirements.txt
streamlit run app.py
```

Windows 可双击 `启动网页版.bat`；命令行扫描使用 `运行扫描.bat` 或：

```bash
python scripts/scan.py --n 12 --depth 300
```

---

## 7. 常见部署错误

| 现象 | 处理 |
|---|---|
| `ModuleNotFoundError: plotly` | 确认 `requirements.txt` 已提交，且没有把 `plotly` 写进 `requirements-dev.txt` 之外 |
| 页面空白/一直转圈 | 查看 Cloud 日志；可能是首次抓行情较慢，或接口被限流 |
| 提示 `data/bootstrap` 缺失 | 本地执行 `python scripts/build_bootstrap.py` 并提交 `data/bootstrap/` |
| 时区/信号日不对 | 程序按 Asia/Shanghai 判断 15:10 收盘确认；云端 UTC 不影响 |
| 表格/图表在手机上太窄 | 表格横向滑动；详情页图表建议竖屏查看 |
| 想换端口 | `streamlit run app.py --server.port 8502` |
