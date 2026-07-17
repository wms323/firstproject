import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import v2
from torch.optim.lr_scheduler import StepLR
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transform=v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32,scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset=torchvision.datasets.CIFAR10(root="./data",train=True,download=True,transform=transform)
test_dataset=torchvision.datasets.CIFAR10(root="./data",train=False,download=True,transform=transform)

train_dataloader=DataLoader(train_dataset,batch_size=64,shuffle=True)
test_dataloader=DataLoader(test_dataset,batch_size=64,shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    

    def forward(self, x):
        x=self.layers(x)
        return x

net=Net()
net=net.to(device)
loss_fn=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
scheduler=StepLR(optimizer,step_size=2,gamma=0.5)

net.train()
num_epochs=10
best_acc=0.0
writer=SummaryWriter("./logs")
for epoch in range(num_epochs):
    running_loss=0.0

    for i,data in enumerate(train_dataloader):
        inputs,labels=data
        inputs,labels=inputs.to(device),labels.to(device)
        outputs=net(inputs)
        loss=loss_fn(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()

    net.eval()
    correct=0
    total=0
    with torch.no_grad():
        for data in test_dataloader:
            inputs,labels=data
            inputs,labels=inputs.to(device),labels.to(device)
            outputs=net(inputs)
            _,predicted=torch.max(outputs,1)
            total+=labels.size(0)
            correct+=predicted.eq(labels).sum().item()
    accuracy=100*correct/total
    net.train()
    writer.add_scalar("Loss/train",running_loss/len(train_dataloader),epoch)
    writer.add_scalar("Accuracy/test",accuracy,epoch)
    writer.add_scalar("LR",optimizer.param_groups[0]['lr'],epoch)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader):.4f}, Accuracy: {accuracy:.2f},LR:{optimizer.param_groups[0]['lr']:.6f}")
    scheduler.step()
    if accuracy>best_acc:
        best_acc=accuracy
        torch.save(net.state_dict(),"best_cifar10.pth")
        print(f"保存最佳模型，准确率：{best_acc:.2f}%")
writer.close()

