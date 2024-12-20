import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridLoss(nn.Module):
    def __init__(self, perceptual_weight=0.8, edge_weight=0.2):
        super(HybridLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.edge_weight = edge_weight

    def forward(self, sr, hr):
        
        print(f"Input shapes - sr: {sr.shape}, hr: {hr.shape}")

       
        min_h = min(sr.size(2), hr.size(2))
        min_w = min(sr.size(3), hr.size(3))

        sr = sr[:, :, :min_h, :min_w]
        hr = hr[:, :, :min_h, :min_w]

      
        print(f"Aligned shapes - sr: {sr.shape}, hr: {hr.shape}")

        
        perceptual_loss = F.l1_loss(sr, hr)

        sr_edges_h = torch.abs(sr[:, :, 1:, :] - sr[:, :, :-1, :])
        sr_edges_w = torch.abs(sr[:, :, :, 1:] - sr[:, :, :, :-1])
        hr_edges_h = torch.abs(hr[:, :, 1:, :] - hr[:, :, :-1, :])
        hr_edges_w = torch.abs(hr[:, :, :, 1:] - hr[:, :, :, :-1])

       
        min_h_edge = min(sr_edges_h.size(2), hr_edges_h.size(2))
        min_w_edge = min(sr_edges_w.size(3), hr_edges_w.size(3))

        sr_edges_h = sr_edges_h[:, :, :min_h_edge, :]
        hr_edges_h = hr_edges_h[:, :, :min_h_edge, :]
        sr_edges_w = sr_edges_w[:, :, :, :min_w_edge]
        hr_edges_w = hr_edges_w[:, :, :, :min_w_edge]

        sr_edges = sr_edges_h[:, :, :min_h_edge, :min_w_edge] + sr_edges_w[:, :, :min_h_edge, :min_w_edge]
        hr_edges = hr_edges_h[:, :, :min_h_edge, :min_w_edge] + hr_edges_w[:, :, :min_h_edge, :min_w_edge]

       
        print(f"Final edge tensor shapes - sr_edges: {sr_edges.shape}, hr_edges: {hr_edges.shape}")

        edge_loss = F.l1_loss(sr_edges, hr_edges)

       
        return self.perceptual_weight * perceptual_loss + self.edge_weight * edge_loss
