import os
import subprocess
import json
def set_up_colab(user="testuser1998", key="65d77e656a286b613bceedd3f4a59c75"):
    subprocess.call('mkdir /root/.kaggle/',shell = True)
    token = {'username':user,'key':key}
    with open('/root/.kaggle/kaggle.json', 'w') as file:
        json.dump(token, file)
    cmds =  ["mkdir data saved_models data/all_images",
             "kaggle competitions download -c pg-image-moderation -p data/",
             "unzip data/all_images.zip -d data/all_images/",
             "rm data/all_images.zip"]
    for cmd in cmds:
        print(subprocess.call(cmd,shell=True),end=' ')
    
if __name__ == '__main__':
    set_up_colab()