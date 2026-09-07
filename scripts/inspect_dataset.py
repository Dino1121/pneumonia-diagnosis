from pathlib import Path
from collections import defaultdict, Counter
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "chest_xray"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# 클래스 판별
def get_class_name(path):

    parts = [part.upper() for part in path.parts]

    if "NORMAL" in parts:
        return "NORMAL"

    if "PNEUMONIA" in parts:
        return "PNEUMONIA"

    return "UNKNOWN"


# 기존 split
def get_split_name(path):

    parts = [part.lower() for part in path.parts]

    if "train" in parts:
        return "train"

    if "val" in parts:
        return "val"

    if "test" in parts:
        return "test"

    return "unknown"


# 파일명에서 group id 추출
def extract_group_id(filename, class_name):
    """
    주의:
    Kaggle v2의 파일명은 irregular하므로
    여기서 추출하는 ID를 확정된 환자 ID라기보다는
    filename-derived group ID로 사용한다.

    PNEUMONIA 예시

    person1_bacteria_1.jpeg
        -> person1

    person1_virus_100.jpeg
        -> person1


    NORMAL 예시

    IM-0011-0001.jpeg
        -> im-0011

    IM-0011-0001-0002.jpeg
        -> im-0011

    NORMAL2-IM-0011-0001.jpeg
        -> im-0011


    동일한 숫자 ID가 여러 filename namespace에서
    나타날 경우에도 하나의 group으로 묶는다.

    실제로 서로 다른 환자라면 단지 같은 split에
    함께 배정될 뿐이지만,

    실제 동일 환자인데 서로 다른 group으로 분리하면
    train / val / test 사이 leakage가 발생할 수 있기 때문이다.
    """

    filename = filename.lower()

    # PNEUMONIA
    if class_name == "PNEUMONIA":

        match = re.search(r"(person\d+)", filename)

        if match:
            return match.group(1)

    # NORMAL
    elif class_name == "NORMAL":

        match = re.search(r"(im-\d+)", filename)

        if match:
            return match.group(1)

    return None


# 전체 이미지 검색
image_files = [

    file

    for file in DATASET_PATH.rglob("*")

    if file.is_file()
    and file.suffix.lower() in IMAGE_EXTENSIONS

]


print("=" * 75)
print("CHEST X-RAY DATASET INSPECTION")
print("=" * 75)

print(f"\n전체 이미지 수: {len(image_files):,}장")


# 통계 저장
class_counter = Counter()
split_counter = Counter()

group_images = defaultdict(list)

group_class = {}
group_splits = defaultdict(set)

unknown_group_files = []

normal_source_groups = defaultdict(set)
pneumonia_type_groups = defaultdict(set)


# 전체 이미지 분석하기
for image_path in image_files:

    class_name = get_class_name(image_path)

    split_name = get_split_name(image_path)

    class_counter[class_name] += 1

    split_counter[split_name] += 1


    group_id = extract_group_id(

        image_path.name,
        class_name

    )


    if group_id is None:

        unknown_group_files.append(image_path)

        continue


    # NORMAL과 PNEUMONIA의 ID가 우연히 같아도
    # 서로 다른 클래스이므로 별개의 그룹으로 취급
    unique_group_id = f"{class_name}_{group_id}"


    group_images[unique_group_id].append(

        image_path

    )


    group_splits[unique_group_id].add(

        split_name

    )


    if unique_group_id not in group_class:

        group_class[unique_group_id] = class_name


    # NORMAL filename namespace 확인용
    if class_name == "NORMAL":

        filename_lower = image_path.name.lower()

        if filename_lower.startswith("normal2-"):

            normal_source_groups["NORMAL2"].add(group_id)

        else:

            normal_source_groups["NORMAL1"].add(group_id)


    # PNEUMONIA subtype namespace 확인용
    elif class_name == "PNEUMONIA":

        filename_lower = image_path.name.lower()

        if "bacteria" in filename_lower:

            pneumonia_type_groups["BACTERIA"].add(group_id)

        elif "virus" in filename_lower:

            pneumonia_type_groups["VIRUS"].add(group_id)


# 클래스별 이미지수
print("\n" + "=" * 75)
print("[클래스별 전체 이미지 수]")
print("=" * 75)

for cls in ["NORMAL", "PNEUMONIA", "UNKNOWN"]:

    count = class_counter.get(cls, 0)

    if count > 0:

        print(f"{cls:12s}: {count:,}장")


# 기존 split별 이미지수
print("\n" + "=" * 75)
print("[기존 Split별 이미지 수]")
print("=" * 75)

for split in ["train", "val", "test", "unknown"]:

    count = split_counter.get(split, 0)

    if count > 0:

        print(f"{split:12s}: {count:,}장")


# group 통계
print("\n" + "=" * 75)
print("[Group 통계]")
print("=" * 75)

print(

    f"식별 가능한 전체 group 수: "
    f"{len(group_images):,}개"

)


group_class_counter = Counter(

    group_class.values()

)


print("\n클래스별 group 수:")

for cls in ["NORMAL", "PNEUMONIA"]:

    print(

        f"{cls:12s}: "
        f"{group_class_counter.get(cls, 0):,}개"

    )


# group당 이미지수
group_sizes = {

    group_id: len(images)

    for group_id, images
    in group_images.items()

}


if group_sizes:

    values = list(group_sizes.values())

    print("\n" + "=" * 75)
    print("[Group당 이미지 수]")
    print("=" * 75)

    print(f"최소: {min(values)}장")

    print(f"최대: {max(values)}장")

    print(

        f"평균: "
        f"{sum(values) / len(values):.2f}장"

    )


# 클래스별 Group 통계
print("\n" + "=" * 75)
print("[클래스별 Group 통계]")
print("=" * 75)


for cls in ["NORMAL", "PNEUMONIA"]:

    sizes = [

        len(group_images[group_id])

        for group_id
        in group_images

        if group_class[group_id] == cls

    ]

    if not sizes:
        continue


    print(f"\n{cls}")

    print(

        f"Group 수       : "
        f"{len(sizes):,}"

    )

    print(

        f"이미지 수      : "
        f"{sum(sizes):,}"

    )

    print(

        f"Group당 평균   : "
        f"{sum(sizes) / len(sizes):.2f}장"

    )

    print(

        f"Group당 최소   : "
        f"{min(sizes)}장"

    )

    print(

        f"Group당 최대   : "
        f"{max(sizes)}장"

    )


# group당 이미지 분포
print("\n" + "=" * 75)
print("[Group당 이미지 개수 분포]")
print("=" * 75)


for cls in ["NORMAL", "PNEUMONIA"]:

    distribution = Counter()

    for group_id, images in group_images.items():

        if group_class[group_id] == cls:

            distribution[len(images)] += 1


    print(f"\n{cls}")

    for image_count in sorted(distribution):

        print(

            f"{image_count:3d}장 보유 group : "
            f"{distribution[image_count]:,}개"

        )


# group당 1장 남긴다는 가정
print("\n" + "=" * 75)
print("[Group당 1장만 남긴다고 가정]")
print("=" * 75)


for cls in ["NORMAL", "PNEUMONIA"]:

    total_images = class_counter.get(cls, 0)

    group_count = group_class_counter.get(cls, 0)

    duplicated_images = (

        total_images - group_count

    )


    print(f"\n{cls}")

    print(

        f"원래 이미지 수         : "
        f"{total_images:,}장"

    )

    print(

        f"Group당 1장 가정       : "
        f"{group_count:,}장"

    )

    print(

        f"추가 반복 이미지 수    : "
        f"{duplicated_images:,}장"

    )


# group중복 체크
overlap_groups = {

    group_id: splits

    for group_id, splits
    in group_splits.items()

    if len(splits) > 1

}


print("\n" + "=" * 75)
print("[기존 Train / Val / Test 사이 Group 중복]")
print("=" * 75)

print(

    f"두 개 이상의 split에 존재하는 group: "
    f"{len(overlap_groups):,}개"

)


if overlap_groups:

    print("\n중복 group 예시 최대 30개:")

    for i, (group_id, splits) in enumerate(

        overlap_groups.items()

    ):

        print(

            f"{group_id:25s} "
            f"-> {sorted(splits)}"

        )

        if i >= 29:

            break


print("\n" + "=" * 75)
print("[이미지가 가장 많은 Group TOP 20]")
print("=" * 75)


sorted_groups = sorted(

    group_images.items(),

    key=lambda x: len(x[1]),

    reverse=True

)


for group_id, images in sorted_groups[:20]:

    print(

        f"{group_id:30s} | "
        f"{len(images):3d}장"

    )


print("\n" + "=" * 75)
print("[Group ID 추출 실패]")
print("=" * 75)

print(

    f"추출 실패 이미지: "
    f"{len(unknown_group_files):,}장"

)


if unknown_group_files:

    print("\n예시 최대 30개:")

    for image_path in unknown_group_files[:30]:

        print(

            get_class_name(image_path),
            "|",
            image_path.name

        )


# NORMAL namespace 검사
normal1_groups = normal_source_groups["NORMAL1"]

normal2_groups = normal_source_groups["NORMAL2"]

normal_namespace_overlap = (

    normal1_groups & normal2_groups

)


print("\n" + "=" * 75)
print("[NORMAL filename namespace 검사]")
print("=" * 75)

print(

    f"NORMAL1 group 수: "
    f"{len(normal1_groups):,}개"

)

print(

    f"NORMAL2 group 수: "
    f"{len(normal2_groups):,}개"

)

print(

    f"두 namespace에서 겹치는 IM ID 수: "
    f"{len(normal_namespace_overlap):,}개"

)


if normal_namespace_overlap:

    print("\n겹치는 ID 예시 최대 20개:")

    for group_id in sorted(

        normal_namespace_overlap

    )[:20]:

        print(group_id)


# PNEUMONIA namespace 검사
bacteria_groups = (

    pneumonia_type_groups["BACTERIA"]

)

virus_groups = (

    pneumonia_type_groups["VIRUS"]

)

pneumonia_namespace_overlap = (

    bacteria_groups & virus_groups

)


print("\n" + "=" * 75)
print("[PNEUMONIA filename namespace 검사]")
print("=" * 75)

print(

    f"BACTERIA group 수: "
    f"{len(bacteria_groups):,}개"

)

print(

    f"VIRUS group 수: "
    f"{len(virus_groups):,}개"

)

print(

    f"BACTERIA / VIRUS에서 겹치는 person ID 수: "
    f"{len(pneumonia_namespace_overlap):,}개"

)


if pneumonia_namespace_overlap:

    print("\n겹치는 ID 예시 최대 20개:")

    for group_id in sorted(

        pneumonia_namespace_overlap

    )[:20]:

        print(group_id)


print("\n" + "=" * 75)
print("SCAN COMPLETE")
print("=" * 75)