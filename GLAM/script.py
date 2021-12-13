for i in range(900//25+1):
    # print(f"python3 main.py --net_g '../output/model_11_30/models/netG_epoch_{i*25}.pth' --cfg 'cfg/eval_bird.yml';")
    # print(f"python3 main.py --net_g '../output/model_11_30/models/netG_epoch_{i*25}.pth' --cfg 'cfg/sample_bird.yml'")
    # print(f"mkdir ../samples/11_30/netG_epoch_{i*25}")
    # print(f"cp -r ../output/model_11_30/models/netG_epoch_{i*25}/Purple_Finch_0025_28174/* ../samples/11_30/netG_epoch_{i*25}")
    print(f"mkdir ../samples/12_01/netG_epoch_{i*25}")
    print(f"cp -r ../output/model_12_01/models/netG_epoch_{i*25}/Purple_Finch_0025_28174/* ../samples/12_01/netG_epoch_{i*25}")