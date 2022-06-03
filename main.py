import os
import json
import time
import schedule
import socket
from datetime import datetime

import torch

from mail import MailSend
from kdTrain import train
from args import get_argparser
import utils


LOGIN = {
    3 : "/mnt/server5/sdi/login.json",
    4 : "/mnt/server5/sdi/login.json",
    5 : "/data1/sdi/login.json"
}
DEFAULT_DIR = {
    3 : "/mnt/server5/sdi",
    4 : "/mnt/server5/sdi",
    5 : "/data1/sdi"
}
DATA_DIR = {
    3 : "/mnt/server5/sdi/datasets",
    4 : "/mnt/server5/sdi/datasets",
    5 : "/data1/sdi/datasets"
}


def smail(subject: str = 'default subject', body: dict = {}, login_dir: str = ''):
    ''' send short report mail (smtp) 
    '''
    # Mail options
    to_addr = ['sdimivy014@korea.ac.kr']
    from_addr = ['singkuserver@korea.ac.kr']

    ms = MailSend(subject=subject, msg=body,
                    login_dir=login_dir,
                    ID='singkuserver',
                    to_addr=to_addr, from_addr=from_addr)
    ms()


if __name__ == '__main__':

    opts = get_argparser().parse_args()

    print('basename:    ', os.path.basename(__file__)) # main.py
    print('dirname:     ', os.path.dirname(__file__)) # /data1/sdi/CPNKD
    print('abspath:     ', os.path.abspath(__file__)) # /data1/sdi/CPNKD/main.py
    print('abs dirname: ', os.path.dirname(os.path.abspath(__file__))) # /data1/sdi/CPNKD

    if socket.gethostname() == "server3":
        opts.cur_work_server = 3
        opts.login_dir = LOGIN[3]
        opts.default_path = os.path.join(DEFAULT_DIR[3], 
                                            os.path.dirname(os.path.abspath(__file__)).split('/')[-1]+'-result')
        opts.data_root = DATA_DIR[3]
    elif socket.gethostname() == "server4":
        opts.cur_work_server = 4
        opts.login_dir = LOGIN[4]
        opts.default_path = os.path.join(DEFAULT_DIR[5], 
                                            os.path.dirname(os.path.abspath(__file__)).split('/')[-1]+'-result')
        opts.data_root = DATA_DIR[4]
    elif socket.gethostname() == "server5":
        opts.cur_work_server = 5
        opts.login_dir = LOGIN[5]
        opts.default_path = os.path.join(DEFAULT_DIR[5], 
                                            os.path.dirname(os.path.abspath(__file__)).split('/')[-1]+'-result')
        opts.data_root = DATA_DIR[5]
    else:
        raise NotImplementedError
    
    if not os.path.exists(opts.login_dir):
        raise FileNotFoundError("login.json file not found (path: {})".format(opts.login_dir))
  
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpus)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    if os.path.exists(os.path.join(opts.default_path, 'log.json')):
        resume = True
        jog = utils.Params(os.path.join(opts.default_path, 'log.json')).__dict__
    else:
        resume = False
        jog = {
                's_model_choice' : 0,
                't_model_choice' : 0,
                't_model_params' : 0,
                'current_working_dir' : 0
                }

    if os.path.exists(os.path.join(opts.default_path, 'mlog.json')):
        mlog = utils.Params(os.path.join(opts.default_path, 'mlog.json')).__dict__
    else:
        mlog = {}

    total_time = datetime.now()
    try:
        opts.Tlog_dir = opts.default_path
        opts.loss_type = 'kd_loss'
        opts.s_model = 'deeplabv3plus_resnet50'
        opts.t_model = 'deeplabv3plus_resnet50'
        opts.t_model_params = '/mnt/server5/sdi/CPNnetV1-result/deeplabv3plus_resnet50/May17_07-37-30_CPN_six/best_param/dicecheckpoint.pt'
        opts.output_stride = 32
        opts.t_output_stride = 32

        if resume and not opts.run_demo:
            resume = False
            logdir = jog['current_working_dir']
            opts.current_time = "resume"
            opts.ckpt = os.path.join(logdir, 'best_param', 'checkpoint.pt')
            resume = False
        elif not resume and not opts.run_demo:
            opts.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            logdir = os.path.join(opts.Tlog_dir, opts.s_model, opts.current_time + '_' + opts.dataset)
        else:
            opts.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            logdir = os.path.join(opts.Tlog_dir, opts.s_model, opts.current_time + '_' + opts.dataset + '_demo')

        # leave log
        with open(os.path.join(opts.default_path, 'log.json'), "w") as f:
            jog['s_model_choice'] = opts.s_model
            jog['t_model_choice'] = opts.t_model
            jog['t_model_params'] = opts.t_model_params
            jog['current_working_dir'] = logdir
            json.dump(jog, f, indent=4)

        start_time = datetime.now()
        mlog['Single experimnet'] = train(devices=device, opts=opts, LOGDIR=logdir)
        time_elapsed = datetime.now() - start_time

        with open(os.path.join(opts.default_path, 'mlog.json'), "w") as f:
            ''' JSON treats keys as strings
            '''
            json.dump(mlog, f, indent=4)
        
        if os.path.exists(os.path.join(logdir, 'summary.json')):
            params = utils.Params(json_path=os.path.join(logdir, 'summary.json')).dict
            params["time_elpased"] = str(time_elapsed)
            utils.save_dict_to_json(d=params, json_path=os.path.join(logdir, 'summary.json'))

        mlog['time elapsed'] = 'Time elapsed (h:m:s.ms) {}'.format(time_elapsed)
        smail(subject="Short report-{}".format("CPN Knowledge distillation"), body=mlog, login_dir=opts.login_dir)
        mlog = {}

        os.remove(os.path.join(opts.default_path, 'mlog.json'))
        os.remove(os.path.join(opts.default_path, 'log.json'))

    except KeyboardInterrupt:
        print("Stop !!!")
        os.remove(os.path.join(opts.default_path, 'log.json'))
    total_time = datetime.now() - total_time

    print('Time elapsed (h:m:s.ms) {}'.format(total_time))