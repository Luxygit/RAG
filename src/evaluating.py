"""evaluating retrieval recall against grounded datasets"""


from typing import Any
from pathlib import Path
from tqdm import tqdm
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
    if (
            not isinstance(student_search_results_path, (str, Path))
            or not isinstance(dataset_path, (str, Path))
            or not str(student_search_results_path).strip()
            or not str(dataset_path).strip()
            ):
        return 0.0
    student_path = Path(student_search_results_path)
    ground_path = Path(dataset_path)
    if not student_path.is_file() or not ground_path.is_file():
        return 0.0
    try:
        with open(student_path, "r", encoding="utf-8") as f:
            student_res = json.load(f)
        with open(ground_path, "r", encoding="utf-8") as f:
            ground_res = json.load(f)
    except Exception as e:
        print(e)
        return 0.0
    correct_count = 0
    found_match = False
    truth_map: dict[str, list[dict[str, Any]]] = {}
    try:
        golden_list = ground_res["rag_questions"]
        student_list = student_res["search_results"]
        for item in golden_list:
            truth_map[item["question_id"]] = item["sources"]
        for st_res in tqdm(student_list, desc="Evaluating recall"):
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
    except Exception as e:
        print(e)
        return 0.0
    if len(golden_list) == 0:
        return 0.0
    recall = correct_count / len(golden_list)
    return recall
