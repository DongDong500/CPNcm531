import subprocess
import schedule
import time
from mail import MailSend

def get_gpu_memory_map() -> str:
    """Get the current gpu usage.

    Return:
        
    """
    output = subprocess.check_output(["nvidia-smi"])
    if isinstance(output, bytes):
        output = output.decode('utf-8')

    return output

def sendmail():

    output = get_gpu_memory_map()
    ms = MailSend(from_addr=['singkuserver@gmail.com'],
                    to_addr=['sdimivy014@korea.ac.kr'],
                    msg=output,
                    login_dir='/data1/sdi/login.json',
                    ID='singkuserver')
    ms()

if __name__ == "__main__":
    '''
    parse nvidia-smi
    '''
    
    schedule.every().day.at("08:30").do(sendmail)
    schedule.every().day.at("18:40").do(sendmail)
    schedule.every().day.at("23:00").do(sendmail)

    while True:
        schedule.run_pending()
        time.sleep(1800)

    