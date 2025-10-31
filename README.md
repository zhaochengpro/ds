### 个人喜欢玩黑箱文化，你们不一样，别上头。

### 配置文件建在策略根目录

### 文件名字.env


## 单向持仓 模式


# 内容


###  DEEPSEEK_API_KEY= 你的deepseek  api密钥

###  BINANCE_API_KEY=

###  BINANCE_SECRET=

###  OKX_API_KEY=

###  OKX_SECRET=

### OKX_PASSWORD=

###  视频教程：https://www.youtube.com/watch?v=Yv-AMVaWUVg


### 准备一台ubuntu服务器 推荐阿里云 香港或者新加坡 轻云服务器


### wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh

### bash Anaconda3-2024.10-1-Linux-x86_64.sh

### source /root/anaconda3/etc/profile.d/conda.sh 
### echo ". /root/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc




### conda create -n ds python=3.10

### conda activate ds

### pip install -r requirements.txt



### apt-get update 更新镜像源


### apt-get upgrade 必要库的一个升级


### apt install npm 安装npm


### npm install pm2 -g 使用npm安装pm2

### conda create -n trail3 python=4.10

## 实时监控看板

### 启动后端与前端

1. 确保 `.env` 中已配置 OKX 和 OpenAI/OpenRouter 相关密钥。
2. （可选）在环境变量中设置需要监控的币种，例如：
   ```bash
   export DASHBOARD_SYMBOLS="BTC ETH SOL"
   ```
   若省略该变量，后端会基于当前持仓或策略记录自动推断。
3. 启动 FastAPI 服务：
   ```bash
   uvicorn dashboard_backend:app --host 0.0.0.0 --port 8000
   ```
4. 浏览器访问 `http://localhost:8000/`，即可查看实时账户收益、持仓和 AI 策略信号。

### API 说明

- `GET /api/overview`：返回账户指标、当前持仓及各币种的策略历史，可通过 `force_refresh=true` 触发从交易所拉取最新数据。
- `GET /api/account`、`/api/positions`、`/api/strategies`：分别获取单独的数据模块。
- `GET /api/health`：快速检查后端状态与配置。
