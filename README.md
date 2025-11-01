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

### MYSQL_HOST=127.0.0.1

### MYSQL_PORT=3306

### MYSQL_USER=your_mysql_user

### MYSQL_PASSWORD=your_mysql_password

### MYSQL_DATABASE=ai_trader

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

1. 在 `.env` 中配置交易所密钥、OpenRouter/DeepSeek 密钥以及 MySQL 连接信息 (`MYSQL_HOST`、`MYSQL_PORT`、`MYSQL_USER`、`MYSQL_PASSWORD`、`MYSQL_DATABASE`)。
2. 安装依赖后执行交易机器人（同时会启动 FastAPI 仪表盘）：
   ```bash
   python ai_trader.py --symbols BTC/USDT ETH/USDT --timeframe 1h --klineNum 200
   ```
3. 浏览器访问 `http://localhost:8000/`，可查看账户指标、实时持仓、AI 信号、资金走势与运行时数据。仪表盘页头会展示最新的运行时长与循环次数。
4. 首次运行会自动初始化所需的 MySQL 表结构，随后账户快照、持仓历史、AI 信号以及运行时指标会滚动写入数据库。

### API 说明

- `GET /api/state`：返回账户资产、当前持仓、策略信号以及聚合的资金走势。
- `GET /api/account`：单独获取账户核心指标。
- `GET /api/positions`：当前持仓列表。
- `GET /api/strategies/batches`：最近的 AI 批量信号。
- `GET /api/strategies/signals`：按币种拆分的信号历史。
- `GET /api/analytics/equity`：从 MySQL 聚合出的日/周/月/年资金走势图数据。
- `GET /api/analytics/runtime`：运行时长、迭代次数及最近循环耗时信息。
