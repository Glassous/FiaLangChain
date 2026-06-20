# FiaLangChain 接入与集成文档

FiaLangChain 是一个独立的 Python AI Agent 微服务，旨在将复杂的 LLM 聊天逻辑、LangChain 框架与高级 Agent（基于 LangGraph）完全封装，对外提供高复用性的流式 AI 服务接口。

---

## 1. 快速启动与部署

### 环境变量配置 (.env)
在项目根目录创建 `.env` 文件，配置所需的 API Key：
```env
# 基础服务配置
PORT=8000
API_TOKEN=your_secure_internal_token

# LLM 供应商密钥
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1

# 联网搜索密钥
BOCHA_API_KEY=your_bocha_key
TAVILY_API_KEY=tvly-...
```

### 本地开发启动
```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 2. API 接口定义

服务采用 **HTTP POST + Server-Sent Events (SSE)** 流式响应协议。

### POST `/api/v1/agent/chat`

#### 请求头 (Headers)
```http
Content-Type: application/json
Authorization: Bearer your_secure_internal_token
```

#### 请求参数 (Body)
```json
{
  "messages": [
    {"role": "user", "content": "今天北京天气怎么样？帮我用计算器算下它的温度乘以10是多少。"}
  ],
  "system_prompt": "你是一个智能助理...",
  "tools": ["weather", "calculator", "web_search"],
  "model_config": {
    "model": "gpt-4o",
    "temperature": 0.7
  }
}
```

#### 响应格式 (SSE Stream)
响应返回 `text/event-stream`。每个事件由 JSON 对象构成：

1. **会话开始** (`type: "start"`)：
   ```json
   {"type": "start"}
   ```
2. **生成规划路径** (`type: "agent_plan"`)：
   ```json
   {
     "type": "agent_plan",
     "data": [
       {"id": 1, "description": "查询北京天气信息", "tool_name": "weather"},
       {"id": 2, "description": "使用计算器计算温度的10倍", "tool_name": "calculator"}
     ]
   }
   ```
3. **规划步骤进度控制** (`type: "plan_item"`)：
   ```json
   {"type": "plan_item", "data": {"index": 0, "status": "in_progress"}}
   ```
4. **工具调用及返回** (`type: "agent_step"`)：
   ```json
   {
     "type": "agent_step",
     "data": {
       "index": 1,
       "tool_name": "weather",
       "tool_input": "{\"location\": \"北京\"}",
       "tool_output": "{\"weather_data\": \"{\\\"location\\\":\\\"北京\\\",\\\"current\\\":{\\\"temp\\\":25}}\"}",
       "err": ""
     }
   }
   ```
5. **思考过程/思维链** (`type: "reasoning"`)：
   ```json
   {"type": "reasoning", "content": "北京现在的温度是 25°C。接下来我需要将这个数值乘以 10..."}
   ```
6. **普通回复 Token** (`type: "token"`)：
   ```json
   {"type": "token", "content": "今天北京的温度是 25°C。"}
   ```
7. **会话结束** (`type: "done"`)：
   ```json
   {"type": "done"}
   ```

---

## 3. Go 语言对接示例

以下是在现有的 Go 业务系统（如 ALChat Web Backend）中如何快速对接并读取流式数据的示例：

```go
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ModelConfig struct {
	Model       string  `json:"model"`
	Temperature float64 `json:"temperature"`
}

type AgentRequest struct {
	Messages     []ChatMessage `json:"messages"`
	SystemPrompt string        `json:"system_prompt"`
	Tools        []string      `json:"tools"`
	ModelConfig  ModelConfig   `json:"model_config"`
}

type SSEEvent struct {
	Type    string      `json:"type"`
	Content string      `json:"content,omitempty"`
	Data    interface{} `json:"data,omitempty"`
}

func CallPythonAgent(ctx context.Context, apiURL, token string, reqBody AgentRequest, eventChan chan<- SSEEvent) error {
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", apiURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("FiaLangChain API error (status %d): %s", resp.StatusCode, string(body))
	}

	reader := bufio.NewReader(resp.Body)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return err
		}

		// 移除换行符并处理 SSE data 标识
		line = bytes.NewBufferString(line).String()
		if len(line) < 6 || line[:6] != "data: " {
			continue
		}

		data := line[6:]
		var event SSEEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}

		eventChan <- event
		if event.Type == "done" {
			break
		}
	}
}
	return nil
}
```

---

## 4. 云服务器部署指南

在生产环境或云服务器（如 Ubuntu / CentOS）上部署 FiaLangChain 微服务，推荐以下两种方式：

### 方案一：使用 Docker Compose 容器化部署 (推荐)

这是最快捷、一致性最高的部署方式。

1. **上传代码**：将 `FiaLangChain` 项目打包上传至云服务器。
2. **准备配置文件**：在服务器的 `FiaLangChain` 根目录下创建生产环境 `.env` 文件：
   ```env
   PORT=8086
   API_TOKEN=your_strong_custom_token
   # 说明：由于重构后密钥均由 Go 业务后端在发起 API 时动态传递，因此此处无需再配置三方 LLM / 搜索的 API Key
   ```
3. **构建并启动容器**：
   ```bash
   docker-compose up --build -d
   ```
4. **验证运行状态**：
   ```bash
   # 检查容器状态
   docker ps
   # 测试服务是否响应
   curl http://localhost:8086/health
   ```

### 方案二：使用 Systemd 进程管理原生部署

如果服务器上不方便安装 Docker，可以使用 Python 虚拟环境配合 Systemd 进行守护进程管理。

1. **安装环境依赖**：
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv -y
   ```
2. **初始化项目与依赖**：
   ```bash
   cd /path/to/FiaLangChain
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **创建 Systemd 服务配置文件**：
   创建并编辑 `/etc/systemd/system/fialangchain.service`：
   ```ini
   [Unit]
   Description=FiaLangChain Agent Service
   After=network.target

   [Service]
   User=root
   WorkingDirectory=/path/to/FiaLangChain
   ExecStart=/path/to/FiaLangChain/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8086
   Restart=always
   EnvironmentFile=/path/to/FiaLangChain/.env

   [Install]
   WantedBy=multi-user.target
   ```
4. **启动并设置开机自启**：
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl start fialangchain
   sudo systemctl enable fialangchain
   ```
5. **查看服务状态与日志**：
   ```bash
   sudo systemctl status fialangchain
   journalctl -u fialangchain -f
   ```

---

## 5. Nginx 反向代理配置 (可选)

为了支持外部安全访问（HTTPS）以及防范超时断连，建议使用 Nginx 反代并配置以下 SSE 流式特殊参数：

```nginx
server {
    listen 80;
    server_name agent.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8086;
        proxy_http_version 1.1;
        
        # 核心：以下配置确保 Server-Sent Events (SSE) 实时流式响应不被 Nginx 缓存缓冲
        proxy_set_header Connection "";
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Host $http_host;
        
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding on;
        
        # 延长超时时间以防止长文本生成时 Nginx 强制断开连接
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }
}
```

