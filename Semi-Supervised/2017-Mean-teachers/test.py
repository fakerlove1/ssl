



for global_step in range(10000):
   alpha = 0.999
   # print(1.0 / float(global_step + 1))
   alpha = min(1.0 - 1.0 / float(global_step + 1), alpha)
   print(alpha)