import matplotlib.pyplot as plt
import numpy as np
import dataclasses
import math
from pycpd import RigidRegistration
from typing import Tuple, Iterable

# Параметры карты и параметры сканирования робота
MAP_SIZE = 15
LIDAR_ANGLE_STEP = 5  # Шаг лидара
LIDAR_ANGLE_RANGE = 360  # Область сканирования лидара в градусах
LIDAR_MAX_DISTANCE = 999  # Максимальная дистанция лидара


@dataclasses.dataclass
class RobotPosition:
    x: float
    y: float
    theta: float


@dataclasses.dataclass
class Room:
    map_as_2d_array: np.ndarray
    map_size: int
    obstacle_positions: Iterable[Tuple[float, float]]


@dataclasses.dataclass
class Transform:
    s: float
    R: np.array
    t: Tuple[float, float]


@dataclasses.dataclass
class RoomScenarioSetup:
    obstacle_positions: Iterable[Tuple[float, float]]
    robot_positions: Iterable[Tuple[RobotPosition, RobotPosition]]



'Комната 1'

obstaclePos1 = [(9, 1), (10, 2), (10, 3), (11, 3), (12, 3), (12, 4), (13, 4), (13, 5), (14, 3)]

position_robot1_start1 = RobotPosition(x=3, y=5, theta=0)
position_robot1_end1 = RobotPosition(x=7, y=9, theta=0)

'Комната 2'

position_robot1_start2 = RobotPosition(x=1, y=11, theta=0)
position_robot1_end2 = RobotPosition(x=7, y=9, theta=0)

'Комната 3'

position_robot1_start3 = RobotPosition(x=6, y=7, theta=0)
position_robot1_end3 = RobotPosition(x=6, y=7, theta=math.radians(30))

'Комната 4'

position_robot1_start4 = RobotPosition(x=6, y=6, theta=0)
position_robot1_end4 = RobotPosition(x=7, y=7, theta=math.radians(30))

'Комната 5'

position_robot1_start5 = RobotPosition(x=8, y=8, theta=0)
position_robot1_end5 = RobotPosition(x=8, y=8, theta=math.radians(180))

scena1 = RoomScenarioSetup(
    obstacle_positions=obstaclePos1,
    robot_positions=[
        (position_robot1_start1, position_robot1_end1),
        (position_robot1_start2, position_robot1_end2),
        (position_robot1_start3, position_robot1_end3),
        (position_robot1_start4, position_robot1_end4),
        (position_robot1_start5, position_robot1_end5),
    ]
)

'Комната 2'

obstaclePos2 = [(4, 8), (10, 8)]


position_robot2_start1 = RobotPosition(x=5, y=3, theta=0)
position_robot2_end1 = RobotPosition(x=8, y=1, theta=0)

position_robot2_start2 = RobotPosition(x=1, y=1, theta=0)
position_robot2_end2 = RobotPosition(x=11, y=1, theta=0)

position_robot2_start3 = RobotPosition(x=6, y=3, theta=math.radians(45))
position_robot2_end3 = RobotPosition(x=6, y=3, theta=math.radians(135))

position_robot2_start4 = RobotPosition(x=5, y=3, theta=0)
position_robot2_end4 = RobotPosition(x=5, y=2, theta=math.radians(30))

position_robot2_start5 = RobotPosition(x=11, y=12, theta=0)
position_robot2_end5 = RobotPosition(x=8, y=12, theta=0)

scena2 = RoomScenarioSetup(
    obstacle_positions=obstaclePos2,
    robot_positions=[
        (position_robot2_start1, position_robot2_end1),
        (position_robot2_start2, position_robot2_end2),
        (position_robot2_start3, position_robot2_end3),
        (position_robot2_start4, position_robot2_end4),
        (position_robot2_start5, position_robot2_end5),
    ]
)

'Комната 3'
obstaclePos3 = [
    (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (5, 4), (5, 3), (5, 2), (5, 1),
    (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (10, 5), (11, 5), (12, 5), (13, 5),
    (1, 10), (2, 10), (3, 10), (4, 10), (5, 10), (5, 11), (5, 12), (5, 13),
    (9, 13), (9, 12), (9, 11), (9, 10), (10, 10), (11, 10), (12, 10), (13, 10)
]

position_robot3_start1 = RobotPosition(x=3, y=7, theta=0)
position_robot3_end1 = RobotPosition(x=10, y=7, theta=0)


position_robot3_start2 = RobotPosition(x=3, y=7, theta=0)
position_robot3_end2 = RobotPosition(x=11, y=7, theta=0)


position_robot3_start3 = RobotPosition(x=7, y=7, theta=math.radians(0))
position_robot3_end3 = RobotPosition(x=7, y=7, theta=math.radians(60))

position_robot3_start4 = RobotPosition(x=7, y=7, theta=math.radians(0))
position_robot3_end4 = RobotPosition(x=7, y=8, theta=math.radians(60))

scena3 = RoomScenarioSetup(
    obstacle_positions=obstaclePos3,
    robot_positions=[
        (position_robot3_start1, position_robot3_end1),
        (position_robot3_start2, position_robot3_end2),
        (position_robot3_start3, position_robot3_end3),
        (position_robot3_start4, position_robot3_end4),
    ]
)

'Комната 4'
obstaclePos4 = [
    (4, 9), (5, 9), (6, 9), (7, 9), (8, 9), (9, 9), (10, 9)
]

position_robot4_start1 = RobotPosition(x=12, y=2, theta=0)
position_robot4_end1 = RobotPosition(x=10, y=4, theta=0)


position_robot4_start2 = RobotPosition(x=2, y=1, theta=0)
position_robot4_end2 = RobotPosition(x=12, y=4, theta=0)


position_robot4_start3 = RobotPosition(x=6, y=6, theta=math.radians(60))
position_robot4_end3 = RobotPosition(x=6, y=6, theta=math.radians(30))


position_robot4_start4 = RobotPosition(x=4, y=4, theta=math.radians(30))
position_robot4_end4 = RobotPosition(x=4, y=4, theta=math.radians(60))

scena4 = RoomScenarioSetup(
    obstacle_positions=obstaclePos4,
    robot_positions=[
        (position_robot4_start1, position_robot4_end1),
        (position_robot4_start2, position_robot4_end2),
        (position_robot4_start3, position_robot4_end3),
        (position_robot4_start4, position_robot4_end4),
    ]
)


def creationRoom(
        room_size: int,
        obstacle_positions: Iterable[Tuple[int, int]],
        robot_pose: RobotPosition,
) -> Room:
    room = np.zeros((room_size, room_size))
    for obstacle_position in obstacle_positions:
        room[obstacle_position[::-1]] = 1

    for i in range(room_size):
        room[(0, i)] = 1
        room[(i, 0)] = 1
        room[(i, room_size - 1)] = 1
        room[(room_size - 1, i)] = 1

    room[(robot_pose.x, robot_pose.y)[::-1]] = 0.5

    return Room(map_as_2d_array=room, map_size=room_size, obstacle_positions=obstacle_positions)


def creationScanMetrics(pos: RobotPosition, room, step=LIDAR_ANGLE_STEP, max_distance=LIDAR_MAX_DISTANCE) -> (
        np.ndarray, np.ndarray):

    'Создание карты стены, робота и получение результата сканирования'

    degreestart = math.degrees(pos.theta)
    angles = np.arange(degreestart, degreestart + LIDAR_ANGLE_RANGE, step) % 360
    scan_distances = []

    for angle in angles:
        rad = np.radians(angle)
        'Проверка если препятствие не обнаружено'
        distance = -1
        for r in range(1, max_distance):
            x = int(pos.x + r * np.cos(rad))
            y = int(pos.y + r * np.sin(rad))

            'Проверка стен и препятствий'
            if x < 0 or y < 0 or x >= MAP_SIZE or y >= MAP_SIZE or room[y, x] == 1:
                distance = r
                break

        scan_distances.append(distance)

    return np.array(scan_distances), angles


def visualize_scan_result(pose: RobotPosition, scan_distances, angles, room, obstacle_positions) -> None:
    """Визуализирует карту, робота, препятствия и результаты сканирования"""
    plt.figure(figsize=(6, 6))
    plt.imshow(room, cmap='Greys', origin='lower')
    plt.scatter(*zip(*obstacle_positions), color='purple', label='Препятствия')
    plt.scatter(pose.x, pose.y, color='green', label='ROBOT', s=100)
    angles = np.radians(angles)

    for i, (angle, distance) in enumerate(zip(angles, scan_distances)):
        if distance != -1:  #
            end_point = (pose.x + distance * np.cos(angle), pose.y + distance * np.sin(angle))
            color = 'red' if i == 0 else 'blue'
            linewidth = 2.0 if i == 0 else 0.5
            plt.plot([pose.x, end_point[0]], [pose.y, end_point[1]], color=color, linewidth=linewidth)

    plt.xlabel('Ось X')
    plt.ylabel('Ось Y')
    plt.title('Robot map with scan result')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


def convertToMassiveDots(scan_distances, angles):
    """
    Конвертирует измерения дистанции и углов в координаты облака точек.н
    """
    angles_rad = np.radians(np.arange(0, LIDAR_ANGLE_RANGE, LIDAR_ANGLE_STEP))

    valid_distances = scan_distances[scan_distances > 0]
    valid_angles_rad = angles_rad[scan_distances > 0]

    x_coordinate = valid_distances * np.cos(valid_angles_rad)
    y_coordinate = valid_distances * np.sin(valid_angles_rad)

    return np.column_stack((x_coordinate, y_coordinate))


def demonstratePoint(A, B, transformed_A=None):

    plt.figure(figsize=(8, 8))

    plt.scatter(A[:, 0], A[:, 1], color='yellow', marker='^', label='Скопление точек 1')

    plt.scatter(B[:, 0], B[:, 1], color='green', marker='o', label='Скопление точек 2')

    if transformed_A is not None:
        plt.scatter(transformed_A[:, 0], transformed_A[:, 1], color='red', marker='x',
                    label='Трансформированое облако точек 1')

    plt.title("Визуализация скопления точек")
    plt.xlabel("Ось X")
    plt.ylabel("Ось Y")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def computePointMassiveTransform(point_cloud1, point_cloud2) -> [np.array, Transform]:

    reg = RigidRegistration(X=point_cloud2, Y=point_cloud1)
    TY, (s_reg, R_reg, t_reg) = reg.register()
    return reg.transform_point_cloud(Y=point_cloud1), Transform(s=s_reg, R=R_reg, t=t_reg)


def displayScanerResult(pose1: RobotPosition, pose2: RobotPosition, calculated_transformation: Transform) -> None:
    PREDICTION = 3

    calculated_cos_theta = np.round(calculated_transformation.R[0, 0], 5)  # округление для избежания ошибок в acos
    estimated_sign_of_theta = 1 if calculated_transformation.R[0, 1] < 0 else -1 if calculated_transformation.R[1, 0] <= 0 else 1

    # Предсказанная по наложению сканов трансформация
    avg_estimated_theta = math.acos(calculated_cos_theta)
    avg_estimated_theta *= estimated_sign_of_theta
    avg_estimated_x = -calculated_transformation.t[0]  # почему "-" ??? :)
    avg_estimated_y = -calculated_transformation.t[1]

    # Истиная трансформация
    true_x_translation = pose2.x - pose1.x
    true_y_translation = pose2.y - pose1.y
    true_theta_translation = pose2.theta - pose1.theta

    # Разница трансформаций
    dx_x = true_x_translation - avg_estimated_x
    dy_y = true_y_translation - avg_estimated_y
    dt_theta = true_theta_translation - avg_estimated_theta

    # Округляем, чтобы не смотреть 20 знаков после плавающей точки
    avg_estimated_theta = np.round(avg_estimated_theta, PREDICTION)
    avg_estimated_x = np.round(avg_estimated_x, PREDICTION)
    avg_estimated_y = np.round(avg_estimated_y, PREDICTION)
    true_x_translation = np.round(true_x_translation, PREDICTION)
    true_y_translation = np.round(true_y_translation, PREDICTION)
    true_theta_translation = np.round(true_theta_translation, PREDICTION)
    dx_x = np.round(abs(dx_x), PREDICTION)
    dy_y = np.round(abs(dy_y), PREDICTION)
    dt_theta = np.round(abs(dt_theta), PREDICTION)

    diff_x_from_room = np.round(dx_x / MAP_SIZE * 100, PREDICTION)
    diff_y_from_room = np.round(dy_y / MAP_SIZE * 100, PREDICTION)
    diff_theta_from_2pi = np.round(dt_theta / math.radians(360) * 100, PREDICTION)

    translation_vector_length = round(
        math.sqrt((dx_x) ** 2 + (dy_y) ** 2), 3
    )
    room_diagonal_length = MAP_SIZE * math.sqrt(2)
    translation_vector_accuracy = round(translation_vector_length / room_diagonal_length * 100, 3)

    print(f"{MAP_SIZE=} {LIDAR_ANGLE_STEP=} {pose1=} {pose2=}")
    print(f"Точность трансформации ветора X/Y - {translation_vector_accuracy}%")

    print(" -- Истинное преобразвание -- ")
    print(f"\tX\t{true_x_translation}")
    print(f"\tY\t{true_y_translation}")
    print(f"\tθ(rad)\t{true_theta_translation}")

    print(" -- Расчитанное преобразование -- ")
    print(f"\tX\t{avg_estimated_x}")
    print(f"\tY\t{avg_estimated_y}")
    print(f"\tθ(rad)\t{avg_estimated_theta}")

    print(" -- Разница преобразований (% от размера комнаты) -- ")
    print(f"\tX\t{dx_x} ({diff_x_from_room}%)")
    print(f"\tY\t{dy_y} ({diff_y_from_room}%)")
    print(f"\tθ(rad)\t{dt_theta} ({diff_theta_from_2pi}%)")


def conductExperimentRun(obstacle_positions, pose1, pose2):
    room = creationRoom(MAP_SIZE, obstacle_positions, pose1)
    scan_distances, angles = creationScanMetrics(pose1, room.map_as_2d_array)
    scan_distances2, angles2 = creationScanMetrics(pose2, room.map_as_2d_array)
    point_cloud1 = convertToMassiveDots(scan_distances, angles)
    point_cloud2 = convertToMassiveDots(scan_distances2, angles2)
    point_cloud1_transformed, transformation = computePointMassiveTransform(point_cloud1, point_cloud2)
    # print(f"{transformation=}")
    visualize_scan_result(pose1, scan_distances, angles, room.map_as_2d_array, room.obstacle_positions)
    visualize_scan_result(pose2, scan_distances2, angles2, room.map_as_2d_array, room.obstacle_positions)
    demonstratePoint(point_cloud1, point_cloud2, point_cloud1_transformed)
    displayScanerResult(pose1, pose2, transformation)
for i, (pose1, pose2) in enumerate(scena1.robot_positions, start=1):
        print(f"  == Комната 1 - Эксперимент {i} ==  ")
        conductExperimentRun(scena1.obstacle_positions, pose1, pose2)

for i, (pose1, pose2) in enumerate(scena2.robot_positions, start=1):
        print(f"  == Комната 2 - Эксперимент {i} ==  ")
        conductExperimentRun(scena2.obstacle_positions, pose1, pose2)

for i, (pose1, pose2) in enumerate(scena3.robot_positions, start=1):
        print(f"  == Комната 3 - Эксперимент {i} ==  ")
        conductExperimentRun(scena3.obstacle_positions, pose1, pose2)

for i, (pose1, pose2) in enumerate(scena4.robot_positions, start=1):
        print(f"  == Комната 4 - Эксперимент {i} ==  ")
        conductExperimentRun(scena4.obstacle_positions, pose1, pose2)
