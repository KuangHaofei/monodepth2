# 2020/2/13 JD
import os

## open
# file_path = "/home/stereye/underwater_ws/monodepth2/train_data"
file_path = "C2_temp"
all_set = os.listdir(file_path)

print(all_set[0])

# split train and val
num_file = len(all_set)
split_sign = int(0.7 * num_file)

train_set = all_set[: split_sign]
val_set = all_set[split_sign: ]

# test_set = all_set

## write
t = open("../splits/omni_uw/train_files.txt", "wb")
v = open("../splits/omni_uw/val_files.txt", "wb")

for i in range(0, split_sign):
    temp = all_set[i].split(".")
    t.write("omni_uw ".encode())
    t.write(temp[0].encode())
    t.write(" l ".encode())
    t.write("\n".encode())
for i in range(split_sign, num_file):
    temp = all_set[i].split(".")
    v.write("omni_uw ".encode())
    v.write(temp[0].encode())
    v.write(" l ".encode())
    v.write("\n".encode())

t.close()
v.close()
