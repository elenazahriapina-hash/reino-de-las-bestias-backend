ANIMAL_RU = {
    "Wolf": {"male": "Волк", "female": "Волчица"},
    "Lion": {"male": "Лев", "female": "Львица"},
    "Tiger": {"male": "Тигр", "female": "Тигрица"},
    "Lynx": {"male": "Рысь", "female": "Рысь"},
    "Panther": {"male": "Пантера", "female": "Пантера"},
    "Bear": {"male": "Медведь", "female": "Медведица"},
    "Fox": {"male": "Койот", "female": "Лиса"},
    "Wolverine": {"male": "Росомаха", "female": "Росомаха"},
    "Deer": {"male": "Олень", "female": "Лань"},
    "Monkey": {"male": "Обезьяна", "female": "Обезьяна"},
    "Rabbit": {"male": "Кролик", "female": "Кролик"},
    "Buffalo": {"male": "Буйвол", "female": "Буйволица"},
    "Ram": {"male": "Баран", "female": "Ибекса"},
    "Capybara": {"male": "Капибара", "female": "Капибара"},
    "Elephant": {"male": "Слон", "female": "Слониха"},
    "Horse": {"male": "Конь", "female": "Лошадь"},
    "Eagle": {"male": "Орёл", "female": "Орлица"},
    "Owl": {"male": "Филин", "female": "Сова"},
    "Raven": {"male": "Ворон", "female": "Ворона"},
    "Parrot": {"male": "Попугай", "female": "Попугаиха"},
    "Snake": {"male": "Змей", "female": "Змея"},
    "Crocodile": {"male": "Крокодил", "female": "Крокодил"},
    "Turtle": {"male": "Черепаха", "female": "Черепаха"},
    "Lizard": {"male": "Ящерица", "female": "Ящерица"},
}


def get_animal_ru_name(animal_code: str, gender: str) -> str:
    g = "female" if gender == "female" else "male"
    return ANIMAL_RU[animal_code][g]


ELEMENT_NUMBER = {
    "Воздух": 1,
    "Вода": 2,
    "Огонь": 3,
    "Земля": 4,
}

FEMALE_SUFFIX_ANIMALS = {"Deer", "Fox", "Lion", "Ram"}


def build_image_key(animal_code: str, element: str, gender: str) -> str:
    element_number = ELEMENT_NUMBER.get(element)
    if not element_number:
        raise ValueError(f"Unsupported element for image key: {element}")
    suffix = "_f" if gender == "female" and animal_code in FEMALE_SUFFIX_ANIMALS else ""
    return f"{animal_code}{element_number}{suffix}"
