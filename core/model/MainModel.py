import torch
from torch import nn

from core.model.Classifier import Classifyer
from core.model.Voter import Voter
from core.model.Encoder import Encoder


class MainModel(nn.Module):
    def __init__(self, args):
        super(MainModel, self).__init__()

        self.args = args
        self.active_modalities = self.parse_modalities(getattr(args, "modalities", "metric,trace,log"))
        print('From Main Model: ', self.active_modalities)
        print('Root loss: ', args.root_loss, 'fault type loss: ',  args.type_loss)
        if args.root_loss == 'focal' or args.root_loss == 'weighted_focal' or args.type_loss == 'focal' or args.type_loss == 'weighted_focal':
            print('Focal gamma: ', args.focal_gamma)

        use_graph_priors = getattr(args, "use_graph_priors", True)

        self.metric_encoder = Encoder(in_dim=args.embedding_dim,
                                      attn_drop=args.attn_drop,
                                      num_heads=args.num_heads,
                                      num_layers=args.num_layers,
                                      graph_hidden_dim=args.graph_hidden,                                
                                      out_dim=args.graph_out,
                                      use_graph_priors=use_graph_priors)
        self.trace_encoder = Encoder(in_dim=args.embedding_dim,
                                      attn_drop=args.attn_drop,
                                      num_heads=args.num_heads,
                                      num_layers=args.num_layers,
                                      graph_hidden_dim=args.graph_hidden,                                
                                      out_dim=args.graph_out,
                                      use_graph_priors=use_graph_priors)
        self.log_encoder = Encoder(in_dim=args.embedding_dim,
                                      attn_drop=args.attn_drop,
                                      num_heads=args.num_heads,
                                      num_layers=args.num_layers,
                                      graph_hidden_dim=args.graph_hidden,                                
                                      out_dim=args.graph_out,
                                      use_graph_priors=use_graph_priors)
        fuse_dim = 3 * args.graph_out

        self.locator = Voter(fuse_dim, 
                                  hiddens=args.linear_hidden,
                                  out_dim=args.N_I)
        self.typeClassifier = Classifyer(in_dim=fuse_dim,
                                         hiddens=args.linear_hidden,
                                         out_dim=args.N_T)
    @staticmethod
    def parse_modalities(modalities_arg):
        alias_map = {
            "metric": "metric",
            "metrics": "metric",
            "trace": "trace",
            "traces": "trace",
            "log": "log",
            "logs": "log",
        }
        active = {
            alias_map[item.strip().lower()]
            for item in str(modalities_arg).split(",")
            if item.strip().lower() in alias_map
        }
        if not active:
            raise ValueError("No valid modalities were provided. Use a subset of: metric, trace, log.")
        return active

    def zero_representation(self, feat):
        return feat.new_zeros((feat.shape[0], self.args.graph_out))

    @staticmethod
    def zero_path_data(path_data):
        return path_data.new_zeros(path_data.shape)

    def forward(self, metric_feat, trace_feat, log_feat, in_degree, out_degree, dist, path_data, attn_mask=None):
        # x_m = metric_feat if "metric" in self.active_modalities else torch.zeros_like(metric_feat)
        # x_t = trace_feat if "trace" in self.active_modalities else torch.zeros_like(trace_feat)
        # x_l = log_feat if "log" in self.active_modalities else torch.zeros_like(log_feat)
        
        # f_m = self.metric_encoder(x_m, in_degree, out_degree, attn_mask, path_data, dist)
        # f_t = self.trace_encoder(x_t, in_degree, out_degree, attn_mask, path_data, dist)
        # f_l = self.log_encoder(x_l, in_degree, out_degree, attn_mask, path_data, dist)
        zero_path_data = self.zero_path_data(path_data)

        if "metric" in self.active_modalities:
            f_m = self.metric_encoder(metric_feat, in_degree, out_degree, attn_mask, zero_path_data, dist)
        else:
            f_m = self.zero_representation(metric_feat)

        if "trace" in self.active_modalities:
            f_t = self.trace_encoder(trace_feat, in_degree, out_degree, attn_mask, path_data, dist)
        else:
            f_t = self.zero_representation(trace_feat)

        if "log" in self.active_modalities:
            f_l = self.log_encoder(log_feat, in_degree, out_degree, attn_mask, zero_path_data, dist)
        else:
            f_l = self.zero_representation(log_feat)

        f = torch.cat((f_m, f_t, f_l), dim=1)
        # failure type identification
        type_logit = self.typeClassifier(f)
        # root cause localization
        root_logit = self.locator(f)

        return (f_m, f_t, f_l), root_logit, type_logit