# Offline model import (without huggingface)

1.  create directory for Your model in vampnet/models/vampnet/loras/name_of_your_model  
2.  From vapnet/runs/name_of_your_model/c2f/best/vampnet and from vapnet/runs/name_of_your_model/coarse/best/vampnet copy files weights.pth and paste them to directory mentioned in first step.  
3.  Change names of files respectively to c2f.pth and coarse.pth  
4.
	a.  If You have access to GUI in Your machine and You want using an app go to: vampnet/vapmnet/interface.py and add name of Your model in method available_models(cls). (like return list_finetuned() + ["default"] + ["guitar_echo_75"])  
It's quite convenient because You can listen to the generated track.  
	b. If not, You can use my draft python file (02_get_tokens.py). I copied parts of source code to generate output tokens. In 12th line You have to type the name of Your model instead my 'vocal_echo_ch'. Place the file in vampnet directory (with app.py and setup.py files)





