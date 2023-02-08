from utils.utils import getOverlap


def EDER(prediction, id, duration, emotion, window_length, stride):
    lol = []
    for i in range(len(prediction)):
        start = stride*i
        end = start + window_length
        lol.append([id, start, end, prediction[i]])
    
    # Merge and split in 2 simple steps: (i) Merge sseg of same speakers then (ii) split different speakers
    # Step 1: Merge adjacent sub-segments that belong to same speaker (or cluster)
    lol = merge_ssegs_same_speaker_adjacent(lol)
    lol = distribute_overlap(lol)
    # write_rttm(lol, out_rttm_file)
    ref = reference_to_lol(id, duration, emotion)
    
    good_preds = 0
    for i in ref:
        candidates = [element for element in lol if element[3]==i[3]]
        ref_interval = [i[1], i[2]]

        for candidate in candidates:
            overlap = getOverlap(ref_interval, [candidate[1], candidate[2]])
            good_preds += overlap
    return 1 - good_preds/duration


def is_overlapped(end1, start2):
    """Returns True if segments are overlapping.
    Arguments
    ---------
    end1 : float
        End time of the first segment.
    start2 : float
        Start time of the second segment.
    Returns
    -------
    overlapped : bool
        True of segments overlapped else False.
    Example
    -------
    >>> from speechbrain.processing import diarization as diar
    >>> diar.is_overlapped(5.5, 3.4)
    True
    >>> diar.is_overlapped(5.5, 6.4)
    False
    """
    if start2 > end1:
        return False
    else:
        return True


def merge_ssegs_same_speaker(lol):
    """Merge sub-segs from the same speaker.
    Arguments
    ---------
    lol : list of list
        Each list contains [rec_id, sseg_start, sseg_end, spkr_id].
    Returns
    -------
    new_lol : list of list
        new_lol contains adjacent segments merged from the same speaker ID.
    Example
    -------
    >>> from speechbrain.processing import diarization as diar
    >>> lol=[['r1', 5.5, 7.0, 's1'],
    ... ['r1', 6.5, 9.0, 's1'],
    ... ['r1', 8.0, 11.0, 's1'],
    ... ['r1', 11.5, 13.0, 's2'],
    ... ['r1', 14.0, 15.0, 's2'],
    ... ['r1', 14.5, 15.0, 's1']]
    >>> diar.merge_ssegs_same_speaker(lol)
    [['r1', 5.5, 11.0, 's1'], ['r1', 11.5, 13.0, 's2'], ['r1', 14.0, 15.0, 's2'], ['r1', 14.5, 15.0, 's1']]
    """
    preds = list(set([i[3] for i in lol]))

    lol_seperated = []
    for i in preds:
        elements_per_pred = [element for element in lol if element[3]==i]
        elements_per_pred.sort(key=lambda x: float(x[1]))
        lol_seperated.append(elements_per_pred)
        
    new_lol = []
    
    for lol_ in lol_seperated:
        # Start from the first sub-seg
        sseg = lol_[0]
        flag = False
        for i in range(1, len(lol_)):
            next_sseg = lol_[i]
            # IF sub-segments overlap AND has same speaker THEN merge
            if is_overlapped(sseg[2], next_sseg[1]) and sseg[3] == next_sseg[3]:
                sseg[2] = next_sseg[2]  # just update the end time
                # This is important. For the last sseg, if it is the same speaker the merge
                # Make sure we don't append the last segment once more. Hence, set FLAG=True
                if i == len(lol_) - 1:
                    flag = True
                    new_lol.append(sseg)
            else:
                new_lol.append(sseg)
                sseg = next_sseg
        # Add last segment only when it was skipped earlier.
        if flag is False:
            new_lol.append(lol_[-1])
    new_lol.sort(key=lambda x: float(x[1]))
    return new_lol


def merge_ssegs_same_speaker_adjacent(lol):
    """Merge adjacent sub-segs from the same speaker.
    Arguments
    ---------
    lol : list of list
        Each list contains [rec_id, sseg_start, sseg_end, spkr_id].
    Returns
    -------
    new_lol : list of list
        new_lol contains adjacent segments merged from the same speaker ID.
    Example
    -------
    >>> from speechbrain.processing import diarization as diar
    >>> lol=[['r1', 5.5, 7.0, 's1'],
    ... ['r1', 6.5, 9.0, 's1'],
    ... ['r1', 8.0, 11.0, 's1'],
    ... ['r1', 11.5, 13.0, 's2'],
    ... ['r1', 14.0, 15.0, 's2'],
    ... ['r1', 14.5, 15.0, 's1']]
    >>> diar.merge_ssegs_same_speaker(lol)
    [['r1', 5.5, 11.0, 's1'], ['r1', 11.5, 13.0, 's2'], ['r1', 14.0, 15.0, 's2'], ['r1', 14.5, 15.0, 's1']]
    """
    new_lol = []
    
    # Start from the first sub-seg
    sseg = lol[0]
    flag = False
    for i in range(1, len(lol)):
        next_sseg = lol[i]
        # IF sub-segments overlap AND has same speaker THEN merge
        if is_overlapped(sseg[2], next_sseg[1]) and sseg[3] == next_sseg[3]:
            sseg[2] = next_sseg[2]  # just update the end time
            # This is important. For the last sseg, if it is the same speaker the merge
            # Make sure we don't append the last segment once more. Hence, set FLAG=True
            if i == len(lol) - 1:
                flag = True
                new_lol.append(sseg)
        else:
            new_lol.append(sseg)
            sseg = next_sseg
    # Add last segment only when it was skipped earlier.
    if flag is False:
        new_lol.append(lol[-1])

    return new_lol


def distribute_overlap(lol):
    """Distributes the overlapped speech equally among the adjacent segments
    with different speakers.
    Arguments
    ---------
    lol : list of list
        It has each list structure as [rec_id, sseg_start, sseg_end, spkr_id].
    Returns
    -------
    new_lol : list of list
        It contains the overlapped part equally divided among the adjacent
        segments with different speaker IDs.
    Example
    -------
    >>> from speechbrain.processing import diarization as diar
    >>> lol = [['r1', 5.5, 9.0, 's1'],
    ... ['r1', 8.0, 11.0, 's2'],
    ... ['r1', 11.5, 13.0, 's2'],
    ... ['r1', 12.0, 15.0, 's1']]
    >>> diar.distribute_overlap(lol)
    [['r1', 5.5, 8.5, 's1'], ['r1', 8.5, 11.0, 's2'], ['r1', 11.5, 12.5, 's2'], ['r1', 12.5, 15.0, 's1']]
    """
    new_lol = []
    sseg = lol[0]
    # Add first sub-segment here to avoid error at: "if new_lol[-1] != sseg:" when new_lol is empty
    # new_lol.append(sseg)
    if len(lol) == 1:
        return lol

    for i in range(1, len(lol)):
        next_sseg = lol[i]
        # No need to check if they are different speakers.
        # Because if segments are overlapped then they always have different speakers.
        # This is because similar speaker's adjacent sub-segments are already merged by "merge_ssegs_same_speaker()"
        if is_overlapped(sseg[2], next_sseg[1]):
            # Get overlap duration.
            # Now this overlap will be divided equally between adjacent segments.
            overlap = sseg[2] - next_sseg[1]
            # Update end time of old seg
            sseg[2] = sseg[2] - (overlap / 2.0)
            # Update start time of next seg
            next_sseg[1] = next_sseg[1] + (overlap / 2.0)
            if len(new_lol) == 0:
                # For first sub-segment entry
                new_lol.append(sseg)
            else:
                # To avoid duplicate entries
                if new_lol[-1] != sseg:
                    new_lol.append(sseg)
            # Current sub-segment is next sub-segment
            sseg = next_sseg
        else:
            # For the first sseg
            if len(new_lol) == 0:
                new_lol.append(sseg)
            else:
                # To avoid duplicate entries
                if new_lol[-1] != sseg:
                    new_lol.append(sseg)
            # Update the current sub-segment
            sseg = next_sseg

    # Add the remaining last sub-segment
    new_lol.append(next_sseg)
    return new_lol


def reference_to_lol(id, duration, emotion):
    """change reference to a list of list
    Arguments
    ---------
    lod : list of dictionary

    Returns
    -------
    lol : list of list
        It has each list structure as [rec_id, sseg_start, sseg_end, spkr_id].
    """
    assert len(emotion) == 1, "NotImplementedError: The solution is only implemented for one-emotion utterance for now."
    lol = []

    start = emotion[0]["start"]
    end = emotion[0]["end"]
    if start > 0:
        lol.append([id, 0, start, "n"])
    lol.append([id, start, end, emotion[0]["emo"][0]])
    if end < duration:
        lol.append([id, end, duration, "n"])
    return lol


if __name__ == "__main__":
    reference = {
        "wav": "/rd_storage/yingzhi_koios/emotion_datasets/ESD/combined/0012/0012_000245_001106_000293.wav",
        "duration": 1.42,
        "emotion": [
            {
                "emo": "sad",
                "start": 0.38,
                "end": 1.42
            }
        ]
    }
    prediction = ["n", "n", "s", "s", "s", "s", "s", "s"]
    EDER(prediction, "ghj", reference["duration"], reference["emotion"], 0.2, 0.2)