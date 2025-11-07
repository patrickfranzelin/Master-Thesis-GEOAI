#!/usr/bin/env python3
import os, requests, time

API_KEY = os.environ.get("RUNPOD_API_KEY")
POD_ID = os.environ.get("RUNPOD_POD_ID")  # e.g. gwjjc11cyg346z

headers = {"Authorization": f"Bearer {API_KEY}"}

def start_pod():
    r = requests.post(f"https://api.runpod.io/v2/pod/{POD_ID}/start", headers=headers)
    print("Start request:", r.status_code, r.text)

def stop_pod():
    r = requests.post(f"https://api.runpod.io/v2/pod/{POD_ID}/stop", headers=headers)
    print("Stop request:", r.status_code, r.text)

def wait_until_ready():
    for _ in range(30):
        r = requests.get(f"https://api.runpod.io/v2/pod/{POD_ID}", headers=headers)
        state = r.json()["data"]["pod"]["desiredStatus"]
        print("Pod state:", state)
        if state == "RUNNING":
            print("✅ Pod is running!")
            return True
        time.sleep(10)
    print("❌ Timeout waiting for pod.")
    return False

if __name__ == "__main__":
    start_pod()
    wait_until_ready()
