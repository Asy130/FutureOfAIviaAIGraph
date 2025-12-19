
"""
Минимальный тест GraphSAGE - проверка установки и импортов.
"""

print("🧪 Минимальный тест GraphSAGE...")

try:
    # 1. Проверяем основные импорты
    print("1. Проверяю базовые импорты...")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    print("   ✅ torch:", torch.__version__)
    
    # 2. Проверяем torch-geometric
    print("2. Проверяю torch-geometric...")
    from torch_geometric.nn import SAGEConv
    print("   ✅ torch-geometric установлен")
    
    # 3. Проверяем нашу модель
    print("3. Проверяю нашу модель GraphSAGE...")
    
    # Определяем модель прямо здесь для теста
    import sys
    import os
    
    # Добавляем родительскую директорию в путь
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    
    # Пытаемся импортировать
    try:
        from models.graphsage_linkpred import GraphSAGELinkPred
        print("   ✅ Модель GraphSAGELinkPred загружена")
    except ImportError as e:
        print(f"   ❌ Ошибка импорта: {e}")
        print("   Создаю модель напрямую...")
        
        # Определяем модель прямо в скрипте
        from torch.nn import Linear, ModuleList
        
        class SimpleGraphSAGE(nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels):
                super().__init__()
                self.conv1 = SAGEConv(in_channels, hidden_channels)
                self.conv2 = SAGEConv(hidden_channels, out_channels)
                self.lin = Linear(2 * out_channels, 1)
            
            def encode(self, x, edge_index):
                x = self.conv1(x, edge_index).relu()
                x = self.conv2(x, edge_index)
                return x
            
            def decode(self, z, edge_label_index):
                edge_features = torch.cat([z[edge_label_index[0]], z[edge_label_index[1]]], dim=-1)
                return self.lin(edge_features).view(-1)
            
            def forward(self, x, edge_index, edge_label_index):
                z = self.encode(x, edge_index)
                return self.decode(z, edge_label_index)
        
        GraphSAGELinkPred = SimpleGraphSAGE
        print("   ✅ Простая модель создана напрямую")
    
    # 4. Создаем синтетические данные
    print("4. Создаю синтетические данные...")
    
    # Случайный граф
    num_nodes = 20
    num_edges = 30
    x = torch.randn(num_nodes, 16)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_label_index = torch.randint(0, num_nodes, (2, 10))
    
    # 5. Тестируем модель
    print("5. Тестирую прямую передачу...")
    model = GraphSAGELinkPred(in_channels=16, hidden_channels=32, out_channels=16)
    
    with torch.no_grad():
        output = model(x, edge_index, edge_label_index)
        print(f"   ✅ Прямая передача работает!")
        print(f"   Размер выхода: {output.shape}")
        print(f"   Диапазон значений: [{output.min():.4f}, {output.max():.4f}]")
    
    # 6. Проверяем метрики
    print("6. Проверяю метрики...")
    try:
        from utils.metrics import calculate_auc_roc
        print("   ✅ Модуль metrics загружен")
        
        # Тестовые данные для AUC
        pos_scores = torch.randn(10)
        neg_scores = torch.randn(10)
        auc = calculate_auc_roc(pos_scores, neg_scores)
        print(f"   Тестовый AUC: {auc:.4f}")
    except ImportError:
        print("   ⚠️ Модуль metrics не найден, пропускаю...")
    
    
    
except Exception as e:
    print(f"\n❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
   
