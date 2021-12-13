for i in range(900//25+1):
    print(f"python3 main.py --net_g '../output/model_12_01/models/netG_epoch_{i*25}.pth' --cfg 'cfg/eval_bird_bert.yml';")
    # print(f"python3 main.py --net_g '../output/model_12_01/models/netG_epoch_{i*25}.pth' --cfg 'cfg/sample_bird_bert.yml';")