from datetime import datetime, timedelta
import re
import subprocess
import time
import random
from contextlib import contextmanager
import asyncio
import os

from .jobArgs import JobArgs
from .jobargs import *
from .jobListen import JobTaskListen
from ..tools import decorated_print, get_logger

logger = get_logger(__name__)

__ALL__ = ["JobTask"]

class JobTask:
    lock_file_path = "job.lock"

    def __init__(self, args, mode=None):
        if isinstance(args, JobArgs):
            self.args = args
        elif isinstance(args, dict):
            self.mode = mode
            self.args = job_args_map[mode](**args)
            
        self.listener = None

    @contextmanager
    def invoke_lock(self):
        try:
            while True:
                if not os.path.exists(self.lock_file_path):
                    break
                wait_time = random.uniform(3, 7)
                print(f">> Lock file exists. Waiting for {wait_time:.2f} seconds...")
                time.sleep(wait_time)

            with open(self.lock_file_path, 'w') as lock_file:
                lock_file.write("Locked")

            yield self

        finally:
            if os.path.exists(self.lock_file_path):
                os.remove(self.lock_file_path)
                print(f">> Lock file removed: {self.lock_file_path}")
        
    async def run_async(self):
        self.args.pre_process()

        cmd = f"jobctl submit {self.args}"
        with self.invoke_lock():
            with self.args.source_framework():
                while True:
                    decorated_print(title="Execute Command", message=cmd)
                    process = subprocess.Popen(
                        cmd, 
                        shell=True, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE, 
                        text=True, 
                        bufsize=1
                    )
                    process.wait()
            
                    stdout = process.stdout.read()
                    stderr = process.stderr.read()
                    if process.returncode != 0:
                        logger.warning(stdout, "\n", stderr)
                        logger.info("Retry submit job task after 3s... ")
                        time.sleep(3)
                    else:
                        break
        
        task_id = re.findall(r'view\?task_id=([0-9a-zA-Z]+)', stderr)[0]
        assert task_id, f"[ERROR] >> task_id not find after submitting."
        # QueuePauser.pause()
        self.listener = JobTaskListen(task_id, self.args.JOB_NAME)
        await self.listener.wait()
        
        self.args.post_process()
        
        return self.listener.status

    def run(self, ignore_failed=False):
        status = asyncio.run(self.run_async())
        if not ignore_failed and status != "success":
            raise RuntimeError(f">> job task status is {status}")
        return status

    @classmethod
    def run_tasks(cls, tasks):
        return JobTaskList(tasks).run()

class JobTaskList(list):
    def __init__(self, tasks=[]):
        assert all(isinstance(task, JobTask) for task in tasks)
        super().__init__(tasks)

    async def run_tasks_async(self):
        coroutine_tasks = [task.run_async() for task in self]
        return await asyncio.gather(*coroutine_tasks)

    def append(self, task):
        assert isinstance(task, JobTask)
        super().append(task)
    
    def run(self, ignore_failed=False):
        statuses = asyncio.run(self.run_tasks_async())
        if not ignore_failed and any(status != "success" for status in statuses):
            raise RuntimeError(f">> job task list statuses are {statuses}")
        return statuses