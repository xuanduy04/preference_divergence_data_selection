from datetime import datetime, timedelta
import hashlib
import time
import aiohttp
import requests
import asyncio

class JobTaskListen:
    status_url = ""
    user_id = ""
    job_api_token = ""
    job_api_secret = ""
    notify_url = ""
    
    listeners, listeners_lock = [], asyncio.Lock()

    @classmethod
    async def listening(cls, check_interval=7):
        status = {
            "running": len(cls.listeners),
            "success": 0,
            "failed": 0,
            "killed": 0
        }
        check_signal = asyncio.Event()
        async def status_display():
            start_timestamp = time.time()
            check_timestamp = time.time() - check_interval
            status_repr = lambda s: " | ".join(f"{k[0].upper()}.{v}" for k, v in s.items())
            while cls.listeners:
                while((time.time() - check_timestamp) < check_interval):
                    await asyncio.sleep(1)
                    wait_time_from_start_listen = str(timedelta(seconds=int(time.time() - start_timestamp)))
                    print((f"\r>> Waiting for job task(s) to complete: [ {status_repr(status)} ] "
                        f"{wait_time_from_start_listen} seconds ..."), end="")
                check_timestamp = time.time()
                check_signal.set()
        
        async def status_check():
            while cls.listeners:
                await check_signal.wait()
                async with cls.listeners_lock:
                    for listener in cls.listeners:
                        try:
                            await listener.fetch_status()
                        except KeyboardInterrupt:
                            raise KeyboardInterrupt
                        except Exception as e:
                            print(">> Request error. Retry later ...")
                            continue
                        if listener.status in ["success", "failed", "killed"]:
                            status["running"] -= 1
                            status[listener.status] += 1
                            cls.listeners.remove(listener)
                check_signal.clear()
        
        display_task = asyncio.create_task(status_display())
        check_task = asyncio.create_task(status_check())
        
        await asyncio.gather(display_task, check_task)
    def __init__(self, task_id, task_name):
        self.listen_timestamp = time.time()
        self.listen_task = task_id
        self.task_name = task_name
        self.status = "start"
    
    @property
    def job_api_secret(self):
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        sign_string = f"{current_time}{self.job_api_token}{self.job_api_secret}"
        md5_hash_sign = hashlib.md5(sign_string.encode()).hexdigest()
        return md5_hash_sign
    
    @property
    def headers(self):
        return {
        }

    async def fetch_status(self):
        get_url = self.status_url.format(task_id=self.listen_task)
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(get_url) as response:
                if response.status == 200:
                    data = await response.json()
                    self.status = data["data"]["task"]["status"]
                else:
                    raise RuntimeError("Requests return code is not 200")

    def status_notify(self, status_list=["success", "failed"]):
        if not self.status in status_list:
            return
        try:
            message = f"[{self.task_name}] task status: {self.status}"
            requests.get(self.notify_url.format(message=message))
        except:
            pass
    
    async def wait(self):
        async with self.listeners_lock:
            self.listeners.append(self)
            if len(self.listeners) == 1:
                asyncio.create_task(self.listening())

        while not self.status in ["success", "failed", "killed"]:
            await asyncio.sleep(7)
    
        self.status_notify()