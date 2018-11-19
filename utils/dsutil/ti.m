function score = TI(prev_frame, cur_frame)
diff_ = cur_frame - prev_frame

std_dev = std(diff_)
