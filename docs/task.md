# Mamba3 多模态交易智能体项目（v5 - 终版架构）

## 任务清单

### Phase 1: Mamba3 核心
- [ ] 项目初始化（目录、依赖）
- [ ] components.py: RMSNorm, SwiGLU, RoPE, SSD
- [ ] mamba3_core.py: Config, InferenceCache, Mamba3

### Phase 2: 多时间框架编码器
- [ ] encoder.py: MultiTimeframeEncoder (6层: 5m/15m/1h/4h/1D/1W) + MarketRegimeDetector

### Phase 3: 消息面感知 ⭐新增
- [ ] news_encoder.py: SentimentEncoder (CryptoBERT + Mamba3 情绪趋势)
- [ ] context_encoder.py: OnChainSentimentEncoder (链上+情绪+事件)
- [ ] fusion.py: CrossModalFusion (门控跨模态注意力)

### Phase 4: 分层决策智能体
- [ ] agent.py: MacroAgent + TacticalAgent + PositionManager + TradeMemory

### Phase 5: 交易环境
- [ ] trading_env.py: 5m 基础 Tick + 实时止盈止损反射弧 + 多模态观察空间 + 分层动作 + 渐进式奖励

### Phase 6: 数据管道
- [ ] fetcher.py: K线 + 资金费率 + 新闻 + 恐惧贪婪 + 多空比 + 清算 + 链上流入
- [ ] features.py: 技术特征 + 消息面特征 + 事件编码
- [ ] dataset.py: 多模态对齐数据加载

### Phase 7: 训练系统
- [ ] pretrain.py: 掩码自监督重建 + 多目标预测 (Uncertainty Weighting)
- [ ] ppo.py: PPO-Lagrangian 安全强化学习 (成本约束)
- [ ] trainer.py: 课程学习与分层冻结调度

### Phase 8: 回测与验证
- [ ] engine.py + metrics.py
- [ ] 测试 + 端到端验证 + 消融实验

### Phase 9: 实盘交易系统与部署 (Live Trading) ⭐新增
- [ ] gateway.py: Binance Futures API 接口整合与拆单逻辑 (TWAP)
- [ ] data_pump.py: WebSocket 实时数据流与 Redis 特征组装器
- [ ] risk_manager.py: 独立安全断路器 (日亏损锁死/频率限制)与双重止损防线
- [ ] inference_node.py: CPU/轻量级 GPU 上的 Mamba3 定时推理服务
- [ ] 部署脚本 (Docker/Systemd) 及沙盒干跑 (Paper Trading) 测试
