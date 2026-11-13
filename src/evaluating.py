""""""


from pathlib import Path
import json


def calculate_iou(start1: int, end1: int, start2: int, end2: int) -> float:
    """calculates the intersection over the union of the two results"""
    end = min(end1, end2)
    start = max(start1, start2)
    intersection = end - start
    if intersection < 0:
        intersection = 0
    chunk1 = end1 - start1
    chunk2 = end2 - start2
    union = chunk1 + chunk2 - intersection
    if union == 0:
        return 0.0
    iou = intersection / union
    return iou


def evaluate(student_search_results_path: Path, dataset_path: Path) -> float:
    """gets both result jsons and compares them using IoU"""
    with open(student_search_results_path, "r", encoding="utf-8") as f:
        student_res = json.load(f)
    with open(dataset_path, "r", encoding="utf-8") as f:
        ground_res = json.load(f)
    correct_count = 0
    found_match = False
    truth_map = {}
    golden_list = ground_res["rag_questions"]
    student_list = student_res["search_results"]
    for item in golden_list:
        truth_map[item["question_id"]] = item["sources"]
    for st_res in student_list:
        st_id = st_res["question_id"]
        if st_id in truth_map:
            found_match = False
            for st_chunk in st_res["retrieved_sources"]:
                for g_chunk in truth_map[st_id]:
                    if (st_chunk["file_path"] == g_chunk["file_path"]
                        and calculate_iou(
                            st_chunk["first_character_index"],
                            st_chunk["last_character_index"],
                            g_chunk["first_character_index"],
                            g_chunk["last_character_index"]
                            ) >= 0.05):
                        found_match = True
                        break
                if found_match is True:
                    correct_count += 1
                    break
    if len(student_list) == 0:
        return 0.0
    recall = correct_count / len(student_list)
    return recall
