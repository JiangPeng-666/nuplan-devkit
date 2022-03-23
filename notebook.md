# DataStructure
- StateSE2 (x, y, heading), in meter

- VehicleParameters: width, height, front_length, rear_length, center_of_gravity

- Box3D: 3D的一个框。提供了可视化方法。
  - width, height, length, yaw
  - label, score
  - velocity, etc

- Point2D:
  - x, y

- StateVector2D：
  - x, y

- FeatureDataType = Union[npt.NDArray[np.float32], torch.Tensor]

- CarFootprint: 车的静态信息
  - oriented_box: OrientedBox 三维的一个框，表示空间位置和大小
    - h, w, l 其中h是空间高度,lw是平面上的长宽
  - rear_axle_to_center_dist: float 尾轴到中心的距离
  - rear_axle: StateSE2 尾轴的位姿
  - get_point_of_interest: List of Point2D 能查询几个关键点的坐标

- DynamicCarState: 车的动态信息
  - speed, acceleration, angular_velocity, angular_acceleration: float, 1D 速度-加速度-角速度-角加速度
  - center_velocity_2d, center_acceleration_2d: StateVector2D, 在中心点的2D 速度-加速度
  - rear_axle_velocity_2d, rear_axle_acceleration_2d: StateVector2D, 尾轴的2D 角速度-角加速度

- EgoState： 自车信息
  - car_footprint: CarFootprint
  - dynamic_car_state: DynamicCarState
  - tire_steering_angle: 轮胎方向角
  - time_point: 时间戳

- Detections: A List of Box3D 表示其他的车

- Raster: Raster模型输入
  - width, height
  - ego_layer 自车图层
  - agents_layer 其他实体图层
  - roadmap_layer 车道图层
  - baseline_paths_layer 车道线图层

- Trajectory: 预测输出
  - position_x, position_y, heading: FeatureDataType, 轨迹各点坐标及方向
  - terminal_position, terminal_heading: FeatureDataType, 最终点坐标及方向



Input:

scenario_builder = build_scenario_builder(cfg)

scenarios = scenario_builder.get_scenarios(scenario_filter, worker)

Dataset = ScenarioDataset(scenarios)

Dataloader = torch.Dataloader(Dataset)

for id, data in enumerate(Dataloader):
  y = model(data)
  visualize(y)