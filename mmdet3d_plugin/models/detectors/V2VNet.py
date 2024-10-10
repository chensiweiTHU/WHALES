# from .OpenCood_detector import OpenCoodDetector
from torch import nn
from .mvx_faster_rcnn import DynamicMVXFasterRCNN, MVXFasterRCNN
from .get_bev import get_bev_FCooper
from mmdet.models import DETECTORS
from types import MethodType
import torch
# import conv GRU
from opencood.models.sub_modules.convgru import ConvGRU
class V2VNetFusion(nn.Module):
    def __init__(self, args):
        super(V2VNetFusion, self).__init__()
        # args = kargs
        in_channels = args['in_channels']
        H, W = args['conv_gru']['H'], args['conv_gru']['W']
        kernel_size = args['conv_gru']['kernel_size']
        num_gru_layers = args['conv_gru']['num_layers']

        self.discrete_ratio = args['voxel_size'][0]
        # self.downsample_rate = args['downsample_rate']
        self.num_iteration = args['num_iteration']
        self.gru_flag = args['gru_flag']
        self.agg_operator = args['agg_operator']

        self.msg_cnn = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3,
                                 stride=1, padding=1)
        self.conv_gru = ConvGRU(input_size=(H, W),
                                input_dim=in_channels * 2,
                                hidden_dim=[in_channels],
                                kernel_size=kernel_size,
                                num_layers=num_gru_layers,
                                batch_first=True,
                                bias=True,
                                return_all_layers=False)
        self.mlp = nn.Linear(in_channels, in_channels)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, (B, C, H, W)
            
        record_len : list
            shape: (B)
            
        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego, 
            shape: (B, L, L, 4, 4) 
            
        Returns
        -------
        Fused feature.
        """
        # x :[x1,x2] x1: b,c,h,w
        new_x = []
        for l in range(self.num_iteration):
            N = len(x)
            for i in range(N):
                neighbor_feature = torch.cat(
                    x, dim=1) #x
                # (N,C,H,W)
                message = self.msg_cnn(neighbor_feature).unsqueeze(0) 
                message = torch.cat([x[i].unsqueeze(0), message], dim=0)
                # (C,H,W)
                if self.agg_operator=="avg":
                    agg_feature = torch.mean(message, dim=0)
                elif self.agg_operator=="max":
                    agg_feature = torch.max(message, dim=0)[0]
                else:
                    raise ValueError("agg_operator has wrong value")
                # (2C, H, W)
                cat_feature = torch.cat(
                    [x[i], agg_feature], dim=1)
                # (C,H,W)
                if self.gru_flag:
                    gru_out = \
                        self.conv_gru(cat_feature.unsqueeze(0))[
                            0][0].squeeze(0)
                else:
                    gru_out = x[i] + agg_feature
                    # gru_out = batch_node_features[b][i, ...] + agg_feature
                    # updated_node_features.append(gru_out.unsqueeze(0))
                new_x.append(gru_out)
            x=new_x
        # (B,C,H,W)
        # out = torch.cat(
        #     [itm[0, ...].unsqueeze(0) for itm in batch_node_features], dim=0)
        out = x[0]
        # (B,C,H,W)
        # out = self.mlp(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) to much memory

        return out

@DETECTORS.register_module(force=True)       
class V2VNet(MVXFasterRCNN):
    def __init__(self, fuse_model, **kwargs):
        super(V2VNet, self).__init__(**kwargs)
        self.fuse_model = V2VNetFusion(fuse_model)
        # BUILD FUSION MODEL
    def fuse():
        pass
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        device = points[0].device
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        if 'cooperative_agents' in img_metas[0].keys():
            cooperative_points = []
            for i in range(len(img_metas)): # we only support 2 agents
                agent = list(img_metas[i]['cooperative_agents'].keys())[0]
                cooperative_point = img_metas[i]['cooperative_agents'][agent]['points']
                if cooperative_point.device != device:
                    cooperative_point = cooperative_point.to(device)
                if len(cooperative_point) == 0:
                    cooperative_point = torch.tensor([[0,0,-1.5,1.3e-3]]).to(device)
                cooperative_points.append(cooperative_point)
        cooperative_img_feats, cooperative_pts_feats = self.extract_feat(
            cooperative_points, img=img, img_metas=img_metas)
        # pts_feats[0] = torch.unsqueeze(pts_feats[0], -1)
        # FUSION
        # Change the fusion model here
        # cooperative_pts_feats[0] = torch.unsqueeze(cooperative_pts_feats[0], -1)
        fused_pts_feats = [pts_feats[0], cooperative_pts_feats[0]]
        fused_pts_feats = self.fuse_model(fused_pts_feats)
        # fused_pts_feats, _ = torch.max(fused_pts_feats, dim=-1)
        fused_pts_feats = [fused_pts_feats]
        # FINISH FUSION
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(fused_pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
        return losses

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        device = points[0].device
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        if 'cooperative_agents' in img_metas[0].keys():
            cooperative_points = []
            for i in range(len(img_metas)): # we only support 2 agents
                agent = list(img_metas[i]['cooperative_agents'].keys())[0]
                cooperative_point = img_metas[i]['cooperative_agents'][agent]['points']
                if cooperative_point.device != device:
                    cooperative_point = cooperative_point.to(device)
                if len(cooperative_point) == 0:
                    cooperative_point = torch.tensor([[0,0,-1.5,1.3e-3]]).to(device)
                cooperative_points.append(cooperative_point)
        cooperative_img_feats, cooperative_pts_feats = self.extract_feat(
            cooperative_points, img=img, img_metas=img_metas)
        fused_pts_feats = [pts_feats[0], cooperative_pts_feats[0]]
        fused_pts_feats = self.fuse_model(fused_pts_feats)
        # fused_pts_feats, _ = torch.max(fused_pts_feats, dim=-1)
        fused_pts_feats = [fused_pts_feats]
        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                fused_pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
                result_dict['img_metas'] = img_metas
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
                result_dict['img_metas'] = img_metas
        return bbox_list