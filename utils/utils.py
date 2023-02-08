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


def write_rttm(segs_list, out_rttm_file):
    """Writes the segment list in RTTM format (A standard NIST format).
    Arguments
    ---------
    segs_list : list of list
        Each list contains [rec_id, sseg_start, sseg_end, spkr_id].
    out_rttm_file : str
        Path of the output RTTM file.
    """

    rttm = []
    rec_id = segs_list[0][0]

    for seg in segs_list:
        new_row = [
            "EMOTION",
            rec_id,
            "0",
            str(round(seg[1], 4)),
            str(round(seg[2] - seg[1], 4)),
            "<NA>",
            "<NA>",
            seg[3],
            "<NA>",
            "<NA>",
        ]
        rttm.append(new_row)

    with open(out_rttm_file, "w") as f:
        for row in rttm:
            line_str = " ".join(row)
            f.write("%s\n" % line_str)
            
