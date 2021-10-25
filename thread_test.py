import time
import threading

def first():
    print("1st")
    time.sleep(1)
    thread2 = threading.Thread(target=second)
    thread2.start()

def second():
    print("2nd")
    time.sleep(3)
    print("3rd")


for i in range(10):
    thread1 = threading.Thread(target=first)
    thread1.start()
    thread1.join()
print("end threads")