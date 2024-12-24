import wandb
import os

# 如果需要设置代理
# os.environ['WANDB_HTTP_PROXY'] = 'http://your-proxy:port'
# os.environ['WANDB_HTTPS_PROXY'] = 'http://your-proxy:port'
# WANDB_HTTP_PROXY=http://lixiangtian:vJ7QF9Sx5qehJlqQga7XVTajpYbPKMG9WrhoURfD3Km8s7VacclbIqlCZnr1@10.1.20.50:23128
# WANDB_HTTPS_PROXY=http://lixiangtian:vJ7QF9Sx5qehJlqQga7XVTajpYbPKMG9WrhoURfD3Km8s7VacclbIqlCZnr1@10.1.20.50:23128
# WANDB_API_KEY=e8ab26345e839fdd0c5ca50a41be0c804bacd820


try:
    print("Testing wandb connection...")
    
    # 使用你的API key登录
    wandb.login(key=os.environ['WANDB_API_KEY'])
    
    # 初始化一个测试项目
    wandb.init(
        project="test-project",
        name="test-run",
        settings=wandb.Settings(init_timeout=300)
    )
    
    # 记录一些数据
    wandb.log({"test": 1})
    
    print("Successfully connected to wandb!")
    
    # 完成
    wandb.finish()

except Exception as e:
    print(f"Failed to connect to wandb: {str(e)}") 