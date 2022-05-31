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

# File Structure

- ./run
  - The core entry.


# Error Summary
- 在十字路口：
  - （跟随错误车道线）应该左转，但直/右转 1.1.1, 1.1.7, 2.1.7, 6.7.2 8.8.3
  - 正确转弯，但未按车道行驶（曲度过大） 6.6.1
  - 正确转弯，但未按车道行驶（曲度过小） 26.4.8
  - 转到一半不再转弯，冲出可行驶区域 2.7.3 20.3.2 23.6.7 30.4.7
  - 被人行横道阻挡 20.8.5
  - 跟车转弯，因为一直原地停留，发生漂移 
- 在分叉口:
  - 明显慢于实际 8.1.1
  - 应当直行，错误转弯 53.3.3
- 在排队等车的时候：
  - 应该原地等待，但从侧面绕过去了 1.2.2, 3.1.8
  - 应该向前移动，但是没动 3.7.6 5.8.1 6.4.1
- 正常行驶时：
  - 明显慢于实际 1.7.3, 1.8.5, 3.1.6, 5.8.8 146.3.1（拥挤时速度偏向保守） 158.7.2
  - 冲出车道，发生碰撞 9.6.2 11.6.1
  - 变道幅度过大，冲出可行驶区域 20.8.2
  
- 弯道行驶:
  - 未按车道线行驶（曲度不够） 3.8.6
  - 明显慢于实际 4.6.2
  - 应当跟车，但是超车 22.7.8
  - 突然转向，发生碰撞 24.4.1

- 其他：
  - 整体而言，轨迹都有向右弯曲的趋势，因此左转的效果不好
  - 速度偏保守，几乎没有快于gt的结果