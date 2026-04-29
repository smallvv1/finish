# 工具箱缺失检测 Web 说明

## 目录结构（已按前后端拆分）

- 后端入口：`backend/app/main.py`
- 后端流程：`backend/app/workflow.py`
- Vue 前端源码：`toolbox-frontend/`
- 旧版静态页面：`frontend/legacy/index.html`
- 配置文件：`config/toolbox_workflow.json`

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 启动后端

在项目根目录执行：

```bash
python backend/app/main.py --host 127.0.0.1 --port 8000
```

## 3. 打开页面

```text
http://127.0.0.1:8000
```

## 4. 前端开发（Vue3 + Vite）

```bash
cd toolbox-frontend
npm install
npm run dev
```

## 5. 前端构建（给后端托管）

```bash
cd toolbox-frontend
npm install
npm run build
```

构建后会生成 `toolbox-frontend/dist`。
后端会优先托管该目录；如果不存在，则自动回退到 `frontend/legacy/index.html`。

## 6. 配置说明

使用 `config/toolbox_workflow.json` 配置模型路径、缺失规则、OCR 参数、运行目录等。
