def getOverlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def get_labels(data, win_len=0.2, stride=0.2):
    emo_list = data["emotion"]
    assert len(emo_list) == 1

    duration = data["duration"]
    emotion = data["emotion"][0]["emo"]
    emo_start = data["emotion"][0]["start"]
    emo_end = data["emotion"][0]["end"]

    number_frames = int(duration / stride) + 1

    intervals = []
    labels = []
    if emo_start != 0:
        intervals.append([0.0, emo_start])
        labels.append("n")
    intervals.append([emo_start, emo_end])
    labels.append(emotion[0])
    if emo_end != duration:
        intervals.append([emo_end, duration])
        labels.append("n")

    start = 0.0
    frame_labels = []
    for i in range(number_frames):
        win_start = start + i * stride
        win_end = win_start + win_len
        
        # make sure that every sample exists in a window
        if win_end >= duration:
            win_end = duration
            win_start = max(duration - win_len, 0)
        
        for j in range(len(intervals)):
            if getOverlap([win_start, win_end], intervals[j]) >= 0.5 * (win_end - win_start):
                emo_frame = labels[j]
                break
        frame_labels.append(emo_frame)
        if win_end >= duration:
            break
    return intervals, labels, frame_labels
                  

def dict_to_rttm(ids, onsets, durations, labels, output_rttm):
    assert len(ids) == len(onsets)
    assert len(onsets) == len(durations)
    assert len(durations) == len(labels)

    with open(output_rttm, 'w') as f:
        for i in range(len(ids)) :
            f.write("SPEAKER " + id[i] + " 1 " + str(onsets[i]) + " " + str(durations[i]) + " <NA> <NA> " + "inference" + " <NA>" + "\n")

