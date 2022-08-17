import torch
import torch.nn.functional as F

class Model_pred:
    def __init__(self, model, dl, tta: bool = True, half: bool = False):
        self.model = model  # model
        self.dl = dl  # dataloader
        self.tta = tta  # tta 적용 flag
        self.half = half  # half 적용 flag (type : floating point 16으로 설정?)

    def __iter__(self):  # 반복자 확인 필요
        self.model.eval()  # 모델 예측모드
        name_list = self.dl.dataset.fnames
        count = 0
        with torch.no_grad():
            for x, y in iter(self.dl):
                x = x.cuda()  # cuda로 데이터
                if self.half: x = x.half()
                p = self.model(x)  # model evalutation
                # detach()는 계산 그래프에서 tensor 분리
                # gradient 계산을 위해 추적할 tensor가 필요하지 않은 경우, 현재 계산 그래프에서 tensor 분리
                py = torch.sigmoid(p).detach()
                if self.tta:  # tta 기법 적용
                    # x, y, xy flips as TTA
                    flips = [[-1], [-2], [-2, -1]]
                    for f in flips:
                        p = self.model(torch.flip(x, f))
                        p = torch.flip(p, f)
                        py += torch.sigmoid(p).detach()
                    py /= (1 + len(flips))
                ## y(Ground Truth)와 py(predict)의 모양이 다른 경우 height과 width를 동일하게 맞춰준다.
                if y is not None and len(y.shape) == 4 and py.shape != y.shape:
                    py = F.upsample(py, size=(y.shape[-2], y.shpe[-1]), mode="bilinear")
                py = py.permute(0, 2, 3, 1).float().cpu()  # permute를 통해 py의 각 차원의 위치 변경 (확인 필요)
                batch_size = len(py)
                for i in range(batch_size):
                    target = y[i].detach().cpu() if y is not None else None
                    yield py[i], target, name_list[count]
                    count += 1

    def __len__(self):
        return len(self.dl.dataset)