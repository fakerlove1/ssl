import numpy as np

with open(r"E:\note\ssl\data\My-ACDC\all.txt", "r") as file:
    all = np.array(file.readlines())
    np.random.shuffle(all)
    val = all[:int(len(all) * 0.2)]
    print(val)
    print(len(val))
    train = all[int(len(all) * 0.2):]

with open(r"E:\note\ssl\data\My-ACDC\train.txt", "a+") as file:
    file.writelines(train)

with open(r"E:\note\ssl\data\My-ACDC\val.txt", "a+") as file:
    file.writelines(val)

# for test in :
#     test=test.strip("\n")
#     print(test)


# print(all)
