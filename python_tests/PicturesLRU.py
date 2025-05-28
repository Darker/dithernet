from typing import Generic, TypeVar, Callable

TImage = TypeVar('TImage')

class PicturesLRU(Generic[TImage]):
    def __init__(self, capacity: int, picture_loader: 'Callable[[str], TImage]') -> None:
        self.capacity = capacity
        self.cache : 'dict[str, TImage]' = {}
        self.order : list[str] = []
        self.picture_loader = picture_loader

    def get(self, image_path: str) -> TImage:
        if image_path in self.cache:
            self.order.remove(image_path)
            self.order.append(image_path)
            return self.cache[image_path]
        else:
            if len(self.cache) >= self.capacity:
                oldest = self.order.pop(0)
                del self.cache[oldest]
            return self._load(image_path)
    
    def _load(self, image_path: str) -> TImage:
        image_data = self.picture_loader(image_path)
        self.cache[image_path] = image_data
        self.order.append(image_path)
        return image_data


