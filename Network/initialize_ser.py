gen_inst_dir = "根据之前的填写地址"
if gen_inst_dir not in sys.path:
    sys.path.append(gen_inst_dir)

from gen_inst import get_serial_port_list, open_serial_port

Using_MODE = "uart"
SER = None

def make_hardware_config():
    if Using_MODE == "uart" :
        # 获取可用的串口的列表
        port_list = get_serial_port_list()

        # 打开列表中的某个串口，进行数据收发
        if port_list:
            # 选择串口
            while True:
                portx_in_port_list = False
                portx = 'COM3'  #input("请输入要打开的串口的名称(例如：COM5)：")
                

                for my_port in port_list:
                    if portx == my_port[:4]:
                        portx_in_port_list = True
                        break
                if portx_in_port_list:
                    break
            # 设置其余参数
            
            bps = 115200
            timeout = 1
            stopbits = 1
            bytesize = 8
            parity = 'Odd'
            # 打开串口
            ser, successful = open_serial_port(portx, bps, timeout, stopbits, bytesize, parity)
            if successful:
                print(f"串口 {portx} 打开成功！")
            else:
                print(f"串口 {portx} 打开失败！")
                raise AssertionError("串口打开失败，程序终止。")
            
            return ser
    
    else:
        return None
        
if __name__ == '__main__':
    SER = make_hardware_config()

    import pickle

    with open("存SER 的绝对地址.pkl", "wb") as f:
        pickle.dump(SER, f)