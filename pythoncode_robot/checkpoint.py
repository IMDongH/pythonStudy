
#!/usr/lib/python2.7
import os
from datetime import datetime
#rpi num : checkpoint_folder
rpi = {'rpi0 ':'9a55593f1f141c367eb9e5405db4b32866b289877313354ffece1ff56f65372f','rpi1 ':1,'rpi2 ':2,'rpi3 ':3}
#file name
current_time = str(datetime.now())
current_time = current_time[0:19]
current_time = current_time.replace("-","")
current_time = current_time.replace(" ","_")
current_time = current_time.replace(":","_")

rpi_list = list(rpi.keys())
rpi_list.sort()
print(rpi_list)
#script
ch_create = 'docker checkpoint create --leave-running=true --checkpoint-dir=/tmp ' + rpi_list[0] + current_time
os.system(ch_create)
os.chdir('/tmp/' + rpi['rpi0 '] + '/checkpoints')
file_name = rpi_list[0][0:4] + "_" + current_time + ".tar"
os.system('tar -cvf ' + file_name + " " + current_time)
#send to server
os.system('scp -i /checkpoint.pem ' + file_name + ' ubuntu@3.87.184.236:' + file_name)
