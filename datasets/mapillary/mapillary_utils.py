from chainer.dataset import download
from collections import namedtuple

root = 'pfnet/chainercv/mapillary'


def get_mapillary():
    """Gets the base path of mapillary dataset.

        Args:
            mode (str): train, validation.
            
        Returns:
            str: Path to the dataset directory.

    """

    data_root = download.get_dataset_directory(root)
    base_path = data_root
    return base_path


Label = namedtuple(
    'Label', ['name', 'id', 'trainId', 'category', 'categoryId',
              'hasInstances', 'ignoreInEval', 'color'])

mapillary_labels = [
    Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('EgoVehicle', 1, 0, 'void', 0, False, False, (0, 0, 0)),
    Label('CarMount', 2, 1, 'void', 0, False, False, (32, 32, 32)),
    Label('WheeledSlow', 3, 2, 'object', 1, True, False, (0, 0, 192)),
    Label('Truck', 4, 3, 'object', 1, True, False, (0, 0, 70)),
    Label('Trailer', 5, 4, 'object', 1, True, False, (0, 0, 110)),
    Label('OtherVehicle', 6, 5, 'object', 1, True, False, (128, 64, 64)),
    Label('OnRails', 7, 6, 'object', 1, False, False, (0, 80, 100)),
    Label('Motorcycle', 8, 7, 'object', 1, True, False, (0, 0, 230)),
    Label('Caravan', 9, 8, 'object', 1, True, False, (0, 0, 90)),
    Label('Car', 10, 9, 'object', 1, True, False, (0, 0, 142)),
    Label('Bus', 11, 10, 'object', 1, True, False, (0, 60, 100)),
    Label('Boat', 12, 11, 'object', 1, True, False, (0, 0, 142)),
    Label('Bicycle', 13, 12, 'object', 1, True, False, (119, 11, 32)),
    Label('TrashCan', 14, 13, 'object', 1, True, False, (180, 165, 180)),
    Label('TrafficSign(Front)', 15, 14, 'object', 1, True, False, (220, 220, 0)),
    Label('TrafficSign(Back)', 16, 15, 'object', 1, True, False, (192, 192, 192)),
    Label('TrafficLight', 17, 16, 'object', 1, True, False, (250, 170, 30)),
    Label('UtilityPole', 18, 17, 'object', 1, True, False, (0, 0, 142)),
    Label('TrafficSignFrame', 19, 18, 'object', 1, True, False, (128, 128, 128)),
    Label('Pole', 20, 19, 'object', 1, True, False, (153, 153, 153)),
    Label('StreetLight', 21, 20, 'object', 1, True, False, (210, 170, 100)),
    Label('Pothole', 22, 21, 'object', 1, False, False, (170, 170, 170)),
    Label('PhoneBooth', 23, 22, 'object', 1, True, False, (0, 0, 142)),
    Label('Manhole', 24, 23, 'object', 1, True, False, (170, 170, 170)),
    Label('Mailbox', 25, 24, 'object', 1, True, False, (33, 33, 33)),
    Label('JunctionBox', 26, 25, 'object', 1, True, False, (40, 40, 40)),
    Label('FireHydrant', 27, 25, 'object', 1, True, False, (100, 170, 30)),
    Label('CCTVCamera', 28, 25, 'object', 1, True, False, (222, 40, 40)),
    Label('CatchBasin', 29, 25, 'object', 1, True, False, (170, 170, 170)),
    Label('Billboard', 30, 25, 'object', 1, True, False, (220, 220, 220)),
    Label('BikeRack', 31, 25, 'object', 1, True, False, (0, 0, 0)),
    Label('Bench', 32, 25, 'object', 1, True, False, (250, 0, 30)),
    Label('Banner', 33, 25, 'object', 1, True, False, (255, 255, 128)),
    Label('Water', 34, 25, 'nature', 2, False, False, (0, 170, 30)),
    Label('Vegetation', 35, 25, 'nature', 2, False, False, (107, 142, 35)),
    Label('Terrain', 36, 25, 'nature', 2, False, False, (152, 251, 152)),
    Label('Snow', 37, 25, 'nature', 2, False, False, (255, 255, 255)),
    Label('Sky', 38, 25, 'nature', 2, False, False, (70, 130, 180)),
    Label('Sand', 39, 25, 'nature', 2, False, False, (128, 64, 64)),
    Label('Mountain', 40, 25, 'nature', 2, False, False, (64, 170, 64)),
    Label('LaneMarking-General', 41, 25, 'marking', 3, False, False, (255, 255, 255)),
    Label('LaneMarking-Crosswalk', 42, 25, 'marking', 3, True, False, (200, 128, 128)),
    Label('OtherRider', 43, 25, 'human', 4, True, False, (255, 0, 0)),
    Label('Motorcyclist', 44, 25, 'human', 4, True, False, (255, 0, 0)),
    Label('Bicyclist', 45, 25, 'human', 4, True, False, (255, 0, 0)),
    Label('Person', 46, 25, 'human', 4, True, False, (220, 20, 60)),
    Label('Tunnel', 47, 25, 'construction', 5, False, False, (150, 120, 90)),
    Label('Building', 48, 25, 'construction', 5, False, False, (70, 70, 70)),
    Label('Bridge', 49, 25, 'construction', 5, False, False, (150, 100, 100)),
    Label('Sidewalk', 50, 25, 'construction', 5, False, False, (244, 35, 232)),
    Label('ServiceLane', 51, 25, 'construction', 5, False, False, (110, 110, 110)),
    Label('Road', 52, 25, 'construction', 5, False, False, (128, 64, 128)),
    Label('RailTrack', 53, 25, 'construction', 5, False, False, (230, 150, 140)),
    Label('PedestrianArea', 54, 25, 'construction', 5, False, False, (96, 96, 96)),
    Label('Parking', 55, 25, 'construction', 5, False, False, (250, 170, 60)),
    Label('CurbCut', 56, 25, 'construction', 5, False, False, (170, 170, 170)),
    Label('Crosswalk-Plain', 57, 25, 'construction', 5, True, False, (140, 140, 200)),
    Label('BikeLane', 58, 25, 'construction', 5, False, False, (128, 64, 255)),
    Label('Wall', 59, 25, 'construction', 5, False, False, (102, 102, 156)),
    Label('Barrier', 60, 25, 'construction', 5, False, False, (102, 102, 156)),
    Label('GuardRail', 61, 25, 'construction', 5, False, False, (180, 165, 180)),
    Label('Fence', 62, 25, 'construction', 5, False, False, (190, 153, 153)),
    Label('Curb', 63, 25, 'construction', 5, False, False, (196, 196, 196)),
    Label('GroundAnimal', 64, 25, 'animal', 6, True, False, (0, 192, 0)),
    Label('Bird', 65, 25, 'animal', 6, True, False, (165, 42, 42)),
]

mapillary_label_names = [
    l.name for l in mapillary_labels if not l.ignoreInEval]

mapillary_label_colors = [
    l.color for l in mapillary_labels if not l.ignoreInEval]