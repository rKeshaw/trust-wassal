from .strategy import Strategy

#import submodlib
import torch
from torch.autograd import Variable
from geomloss import SamplesLoss
from torch.utils.data import DataLoader, Dataset
import random
import math
from torchvision.models import resnet50, resnet18,resnet101
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class customSampler(torch.utils.data.Sampler):
    def __init__(self, ind):
        self.ind = ind
        return
    def __iter__(self):
        return iter(self.ind)

class WASSAL_Multiclass(Strategy):
    
    """
    
    
    Parameters
    ----------
    labeled_dataset: torch.utils.data.Dataset
        The labeled dataset to be used in this strategy. For the purposes of selection, the labeled dataset is not used, 
        but it is provided to fit the common framework of the Strategy superclass.
    unlabeled_dataset: torch.utils.data.Dataset
        The unlabeled dataset to be used in this strategy. It is used in the selection process as described above.
        Importantly, the unlabeled dataset must return only a data Tensor; if indexing the unlabeled dataset returns a tuple of 
        more than one component, unexpected behavior will most likely occur.
    nclasses: int
        The number of classes being predicted by the neural network.
    args: dict
        A dictionary containing many configurable settings for this strategy. Each key-value pair is described below:
            
            - **b_size**: The batch size used internally for torch.utils.data.DataLoader objects. (int, optional)
            - **device**: The device to be used for computation. PyTorch constructs are transferred to this device. Usually is one of 'cuda' or 'cpu'. (string, optional)
            - **loss**: The loss function to be used in computations. (typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional)
            - **optimizer**: The optimizer to use for submodular maximization. Can be one of 'NaiveGreedy', 'StochasticGreedy', 'LazyGreedy' and 'LazierThanLazyGreedy'. (string, optional)
            - **eta**: A magnification constant that is used in all but gcmi. It is used as a value of query-relevance vs diversity trade-off. Increasing eta tends to increase query-relevance while reducing query-coverage and diversity. (float)
            - **embedding_type**: The type of embedding to compute for similarity kernel computation. This can be either 'gradients' or 'features'. (string)
            - **verbose**: Gives a more verbose output when calling select() when True. (bool)
    """
    
    def __init__(self, labeled_dataset, unlabeled_dataset, query_dataset,net, nclasses, args={}): #
        #pretrained resnet18 as
        #self.pretrained_model=True 
        #self.net = resnet18(pretrained=True)
        self.pretrained_model=False 
        self.net=net
        #merge labeled and query dataset into query and non-query classes and finetune the resnet50
        
        
        super(WASSAL_Multiclass, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)        
        self.query_dataset = query_dataset
        self.args['verbose']=True
        self.temp_first_time=True
        self.temp_second_time=False
        
        

    def _proj_simplex(self,v):
        """
        v: PyTorch Tensor to be projected to a simplex

        Returns:
        w: PyTorch Tensor simplex projection of v
        """
        z = 1
        orig_shape = v.shape
        v = v.view(1, -1)
        shape = v.shape
        with torch.no_grad():
            mu = torch.sort(v, dim=1)[0]
            mu = torch.flip(mu, dims=(1,))
            cum_sum = torch.cumsum(mu, dim=1)
            j = torch.unsqueeze(torch.arange(1, shape[1] + 1, dtype=mu.dtype, device=mu.device), 0)
            rho = torch.sum(mu * j - cum_sum + z > 0.0, dim=1, keepdim=True) - 1.
            rho = rho.to(int)
            max_nn = cum_sum[torch.arange(shape[0]), rho[:, 0]]
            theta = (torch.unsqueeze(max_nn, -1) - z) / (rho.type(max_nn.dtype) + 1)
            w = torch.clamp(v - theta, min=0.0).view(orig_shape)
            return w

    def get_query_simplex(self):
        return self.simplex_query
    
    def _compute_features(self, dataset, embedding_type, layer_name=None, gradType=None,isLabeled=False):
        """Helper method to compute features for a dataset."""
        dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
        features = []
         # Ensure model is in evaluation mode for consistent feature extraction
        self.model.eval()
        with torch.no_grad():
            for batch_idx, inputs in enumerate(dataloader):
                if(isLabeled):
                    images, _ = inputs
                else:
                    images = inputs
                if embedding_type == "features":
                    batch_features= self.get_feature_embedding(images, True, layer_name).view(len(images), -1)
                elif embedding_type == "gradients":
                    batch_features= self.get_grad_embedding(images, False, gradType).view(len(images), -1)
                else:
                    raise ValueError("Unknown embedding type.")
                 # Move the batch features to CPU memory to free up GPU memory
                features.append(batch_features.cpu())
        # Stack all the features and move to the desired device, if needed
        features = torch.vstack(features)
        return features
    #update based on we needed AL model or pretrained model
    def update_model(self, clf):
        if not self.pretrained_model:
            self.model = clf

    def select_only_elements(self, budget):
        """
        Selects next set of points based on optimizing Wasserstein distance
        between labeled (divided by class) and unlabeled batches.
        
        Parameters
        ----------
        budget: int
            Number of data points to select for labeling
            
        Returns
        ----------
        idxs: list
            List of selected data point indices with respect to unlabeled_dataset
        """ 
        
        


        embedding_type = self.args.get('embedding_type', "features")
        layer_name = self.args.get('layer_name', "avgpool") if embedding_type == "features" else None
        gradType = self.args.get('gradType', "bias_linear") if embedding_type == "gradients" else None
        minibatch_size = self.args.get('minibatch_size', None)
        lr = self.args.get('lr', 0.001)
        step_size = self.args.get('step_size', 10)
        iterations = self.args.get('wassal_iterations', 100)

        loss_func = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.7, backend="tensorized")

        # Determine the unique classes in the query dataset
        classes = torch.stack([item[1] for item in self.query_dataset]).to(self.device)  # Assuming item[1] is the class label
        unique_classes = torch.unique(classes)

        unlabeled_dataset_len = len(self.unlabeled_dataset)
        if minibatch_size is None:
            # Default to the full pool in one batch, matching the paper's
            # stated design.
            minibatch_size = unlabeled_dataset_len
        shuffled_indices = list(range(unlabeled_dataset_len))
        random.shuffle(shuffled_indices)

        # Initializing optimization
        simplex_tensor = torch.ones(unlabeled_dataset_len, device=self.device) / unlabeled_dataset_len
        simplex_tensor.requires_grad_(True)
        optimizer = torch.optim.Adam([simplex_tensor], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

        # Precompute features on the same device
        query_dataset_features = self._compute_features(self.query_dataset, embedding_type, layer_name, gradType, True).to(self.device)
        unlabeled_dataset_features = self._compute_features(self.unlabeled_dataset, embedding_type, layer_name, gradType, False).to(self.device)

        temp_loop=True
        for epoch in range(iterations):
            total_loss = 0.0
            optimizer.zero_grad()

            num_batches = math.ceil(unlabeled_dataset_len / minibatch_size)
            if num_batches > 1:
                # Merge a pathologically small remainder batch into the
                # previous one instead of leaving it standalone - verified
                # by direct testing that a very small last batch (e.g. size
                # 1) can degenerate geomloss's Sinkhorn computation
                # (ValueError: Maximum allowed size exceeded / divide-by-zero
                # in its epsilon schedule). The existing min(...) end bound
                # in the loop below then absorbs the remainder automatically
                # once num_batches is reduced.
                last_batch_size = unlabeled_dataset_len - (num_batches - 1) * minibatch_size
                if last_batch_size < minibatch_size / 2:
                    num_batches -= 1
            for batch_idx in range(num_batches):
                start_idx = batch_idx * minibatch_size
                end_idx = min((batch_idx + 1) * minibatch_size, unlabeled_dataset_len)
                batch_features = unlabeled_dataset_features[start_idx:end_idx]
                batch_simplex = simplex_tensor[start_idx:end_idx]
                batch_simplex = batch_simplex / batch_simplex.sum()  # Normalize simplex for current batch

                batch_loss = 0.0
                for cls in unique_classes:
                    cls_mask = (classes == cls)
                    cls_features = query_dataset_features[cls_mask]
                    cls_beta = torch.ones(cls_features.size(0), device=self.device) / cls_features.size(0)

                    if cls_features.size(0) > 0:
                        #for debugging purposes:
                        # if temp_loop:
                        #     temp_loop=False
                        #     if self.temp_second_time:
                        #         self.temp_second_time=False
                        #         self.temp_loss2=loss_func(self.temp_batch_simplex, self.temp_batch_features, self.temp_cls_beta, self.temp_cls_features)
                        #         print('loss',self.temp_loss,'loss2',self.temp_loss2)

                        #     if self.temp_first_time:
                        #         self.temp_first_time=False
                        #         self.temp_second_time=True
                        #         self.temp_batch_simplex=batch_simplex
                        #         self.temp_batch_features= batch_features
                        #         self.temp_cls_beta=cls_beta
                        #         self.temp_cls_features=cls_features
                        #         self.temp_loss=loss_func(batch_simplex, batch_features, cls_beta, cls_features)
                            

                        batch_loss += loss_func(batch_simplex, batch_features, cls_beta, cls_features) / num_batches


                total_loss += batch_loss
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                simplex_tensor.data = self._proj_simplex(simplex_tensor.data)  # Project simplex back after update

            print(f"Epoch: [{epoch}], Loss: {total_loss.item():.4f}", end="\r")

            if abs(total_loss.item()) < 0.2 and epoch > self.args.get('min_iteration', 50):
                break

        sorted_values, sorted_indices = torch.sort(simplex_tensor, descending=True)
        selected_indices = sorted_indices[:budget].cpu().numpy()
        
        return selected_indices, None


    def select_elements_and_soft(self, budget):
        """
        Selects next set of points based on optimizing Wasserstein distance
        between labeled (divided by class) and unlabeled batches.
        
        Parameters
        ----------
        budget: int
            Number of data points to select for labeling
            
        Returns
        ----------
        idxs: list
            List of selected data point indices with respect to unlabeled_dataset
        """ 
        
        


        embedding_type = self.args.get('embedding_type', "features")
        layer_name = self.args.get('layer_name', "avgpool") if embedding_type == "features" else None
        gradType = self.args.get('gradType', "bias_linear") if embedding_type == "gradients" else None
        minibatch_size = self.args.get('minibatch_size', None)
        lr = self.args.get('lr', 0.001)
        step_size = self.args.get('step_size', 10)
        iterations = self.args.get('wassal_iterations', 100)

        loss_func = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.7, backend="tensorized")

        # Determine the unique classes in the query dataset
        classes = torch.stack([item[1] for item in self.query_dataset]).to(self.device)  # Assuming item[1] is the class label
        unique_classes = torch.unique(classes)
        num_classes = len(torch.unique(torch.stack([item[1] for item in self.query_dataset])))

        unlabeled_dataset_len = len(self.unlabeled_dataset)
        if minibatch_size is None:
            # Default to the full pool in one batch, matching the paper's
            # stated design.
            minibatch_size = unlabeled_dataset_len
        shuffled_indices = list(range(unlabeled_dataset_len))
        random.shuffle(shuffled_indices)

        # Initializing optimization
        simplex_tensor = torch.ones(unlabeled_dataset_len, device=self.device) / unlabeled_dataset_len
        simplex_tensor.requires_grad_(True)
        optimizer = torch.optim.Adam([simplex_tensor], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

        # Precompute features on the same device
        query_dataset_features = self._compute_features(self.query_dataset, embedding_type, layer_name, gradType, True).to(self.device)
        unlabeled_dataset_features = self._compute_features(self.unlabeled_dataset, embedding_type, layer_name, gradType, False).to(self.device)
        num_batches = math.ceil(unlabeled_dataset_len / minibatch_size)
        if num_batches > 1:
            # Merge a pathologically small remainder batch into the
            # previous one instead of leaving it standalone - verified
            # by direct testing that a very small last batch (e.g. size
            # 1) can degenerate geomloss's Sinkhorn computation
            # (ValueError: Maximum allowed size exceeded / divide-by-zero
            # in its epsilon schedule). The existing min(...) end bound
            # in the loop below then absorbs the remainder automatically
            # once num_batches is reduced.
            last_batch_size = unlabeled_dataset_len - (num_batches - 1) * minibatch_size
            if last_batch_size < minibatch_size / 2:
                num_batches -= 1
        temp_loop=True
        for epoch in range(iterations):
            total_loss = 0.0
            optimizer.zero_grad()

            for cls in unique_classes:
                cls_mask = (classes == cls)
                cls_features = query_dataset_features[cls_mask]
                cls_beta = torch.ones(cls_features.size(0), device=self.device) / cls_features.size(0)
                if cls_features.size(0) > 0:
                    for batch_idx in range(num_batches):
                        start_idx = batch_idx * minibatch_size
                        end_idx = min((batch_idx + 1) * minibatch_size, unlabeled_dataset_len)
                        batch_features = unlabeled_dataset_features[start_idx:end_idx]
                        batch_simplex = simplex_tensor[start_idx:end_idx]
                        batch_simplex = batch_simplex / batch_simplex.sum()  # Normalize simplex for current batch

                        batch_loss = 0.0
                    

                        
                            #for debugging purposes:
                            # if temp_loop:
                            #     temp_loop=False
                            #     if self.temp_second_time:
                            #         self.temp_second_time=False
                            #         self.temp_loss2=loss_func(self.temp_batch_simplex, self.temp_batch_features, self.temp_cls_beta, self.temp_cls_features)
                            #         print('loss',self.temp_loss,'loss2',self.temp_loss2)

                            #     if self.temp_first_time:
                            #         self.temp_first_time=False
                            #         self.temp_second_time=True
                            #         self.temp_batch_simplex=batch_simplex
                            #         self.temp_batch_features= batch_features
                            #         self.temp_cls_beta=cls_beta
                            #         self.temp_cls_features=cls_features
                            #         self.temp_loss=loss_func(batch_simplex, batch_features, cls_beta, cls_features)
                                

                        batch_loss += loss_func(batch_simplex, batch_features, cls_beta, cls_features) / num_batches


                        total_loss += batch_loss
                
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                simplex_tensor.data = self._proj_simplex(simplex_tensor.data)  # Project simplex back after update

            print(f"Epoch: [{epoch}], Eq 5 Loss: {total_loss.item():.4f}", end="\r")

            if abs(total_loss.item()) < 0.2 and epoch > self.args.get('min_iteration', 50):
                break
        

        sorted_values, sorted_indices = torch.sort(simplex_tensor)
        selected_indices = sorted_indices[:budget].cpu().numpy()

        #once pu is found do equ 6 of https://www.overleaf.com/project/6209ee726c723a7ddb95defa

        classwise_simplex_query = []
        
        for _ in range(num_classes):
            tensor = torch.ones(unlabeled_dataset_len, device=self.device)
            tensor = (tensor / unlabeled_dataset_len).clone().detach().requires_grad_(True)
            classwise_simplex_query.append(tensor)

        label_to_simplex_query = {}
        unique_labels = torch.unique(torch.stack([item[1] for item in self.query_dataset]))
        for i, label in enumerate(unique_labels):
            label_to_simplex_query[label.item()] = classwise_simplex_query[i]
        
        optimizer = torch.optim.Adam(classwise_simplex_query, lr=lr)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

        for epoch in range(iterations):
            total_loss = 0.0
            optimizer.zero_grad()

            for cls in unique_classes:
                cls_mask = (classes == cls)
                cls_features = query_dataset_features[cls_mask]
                cls_beta = torch.ones(cls_features.size(0), device=self.device) / cls_features.size(0)
                simplex_query = classwise_simplex_query[cls]
                simplex_query.requires_grad = True
                if cls_features.size(0) > 0:
                    for batch_idx in range(num_batches):
                        start_idx = batch_idx * minibatch_size
                        end_idx = min((batch_idx + 1) * minibatch_size, unlabeled_dataset_len)
                        batch_features = unlabeled_dataset_features[start_idx:end_idx]
                        batch_simplex = simplex_tensor[start_idx:end_idx]
                        batch_simplex = batch_simplex / batch_simplex.sum()  # Normalize simplex for current batch

                        simplex_batch_query = simplex_query[start_idx : end_idx]
                        if(simplex_batch_query.sum()!=0):
                            simplex_batch_query = simplex_batch_query.clone() / simplex_batch_query.sum()
                        #simplex_batch_query.requires_grad = True
                        simplex_batch_query=simplex_batch_query.to(self.device)

                        batch_loss = 0.0
                    

                        
                            #for debugging purposes:
                            # if temp_loop:
                            #     temp_loop=False
                            #     if self.temp_second_time:
                            #         self.temp_second_time=False
                            #         self.temp_loss2=loss_func(self.temp_batch_simplex, self.temp_batch_features, self.temp_cls_beta, self.temp_cls_features)
                            #         print('loss',self.temp_loss,'loss2',self.temp_loss2)

                            #     if self.temp_first_time:
                            #         self.temp_first_time=False
                            #         self.temp_second_time=True
                            #         self.temp_batch_simplex=batch_simplex
                            #         self.temp_batch_features= batch_features
                            #         self.temp_cls_beta=cls_beta
                            #         self.temp_cls_features=cls_features
                            #         self.temp_loss=loss_func(batch_simplex, batch_features, cls_beta, cls_features)
                                

                        batch_loss += (loss_func(simplex_batch_query, batch_features, cls_beta, cls_features)-loss_func(simplex_batch_query, batch_features, batch_simplex, batch_features) )/ num_batches


                        total_loss += batch_loss
                
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                for class_idx in range(num_classes):
                    classwise_simplex_query[class_idx].data = self._proj_simplex(classwise_simplex_query[class_idx].data)

            print(f"Epoch: [{epoch}], Eq 6 Loss: {total_loss.item():.4f}", end="\r")

            if abs(total_loss.item()) < 0.2 and epoch > self.args.get('min_iteration', 50):
                break
            
        output=[]   
        for iteridx,(class_idx, simplex_query) in enumerate(label_to_simplex_query.items()):
            masked_simplex_query = simplex_query.clone()
            for idx in selected_indices:
                masked_simplex_query[idx] = 0

            output.append((masked_simplex_query.detach().cpu(), masked_simplex_query.detach().cpu(),class_idx))

            

        torch.cuda.empty_cache()
        return selected_indices, output

    def select_for_query_refrain(self, budget):
          # venkat sir's code
        """
        Selects next set of points. Weights are all reset since in this 
        strategy the datapoints are removed
        
        Parameters
        ----------
        budget: int
            Number of data points to select for labeling
            
        Returns
        ----------
        idxs: list
            List of selected data point indices with respect to unlabeled_dataset
        """	
        
        #Get hyperparameters from args dict
        embedding_type = self.args['embedding_type'] if 'embedding_type' in self.args else "features"
        if(embedding_type=="features"):
            layer_name = self.args['layer_name'] if 'layer_name' in self.args else "avgpool"
        gradType=None
        if(embedding_type=="gradients"):
            gradType = self.args['gradType'] if 'gradType' in self.args else "bias_linear"
        loss_func = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.7,backend="tensorized")
        
        unlabeled_dataset_len=len(self.unlabeled_dataset)
        shuffled_indices = list(range(unlabeled_dataset_len))
        random.shuffle(shuffled_indices)
        sampler = customSampler(shuffled_indices)

        query_dataset_len = len(self.query_dataset)
        minibatch_size = self.args['minibatch_size'] if 'minibatch_size' in self.args else unlabeled_dataset_len

        num_batches = math.ceil(unlabeled_dataset_len/minibatch_size)
        if num_batches > 1:
            # Merge a pathologically small remainder batch into the
            # previous one instead of leaving it standalone - verified
            # by direct testing that a very small last batch (e.g. size
            # 1) can degenerate geomloss's Sinkhorn computation
            # (ValueError: Maximum allowed size exceeded / divide-by-zero
            # in its epsilon schedule). The existing min(...) end bound
            # in the loop below then absorbs the remainder automatically
            # once num_batches is reduced.
            last_batch_size = unlabeled_dataset_len - (num_batches - 1) * minibatch_size
            if last_batch_size < minibatch_size / 2:
                num_batches -= 1

        num_classes = len(torch.unique(torch.stack([item[1] for item in self.query_dataset])))
        classwise_simplex_query = []
        classwise_simplex_refrain = []
        for _ in range(num_classes):
            tensor = torch.ones(unlabeled_dataset_len, device=self.device)
            tensor = (tensor / unlabeled_dataset_len).clone().detach().requires_grad_(True)
            classwise_simplex_query.append(tensor)
        
        for _ in range(num_classes):
            tensor = torch.ones(unlabeled_dataset_len, device=self.device)
            tensor = (tensor / unlabeled_dataset_len).clone().detach().requires_grad_(True)
            classwise_simplex_refrain.append(tensor)



        label_to_simplex_query = {}
        unique_labels = torch.unique(torch.stack([item[1] for item in self.query_dataset]))
        
        for i, label in enumerate(unique_labels):
            label_to_simplex_query[label.item()] = classwise_simplex_query[i]
            

         # Hyperparameters for sinkhorn iterations
         #if self.args has lr, use that else use 0.001
        lr = self.args['lr'] if 'lr' in self.args else 0.001
        min_iteration=self.args['min_iteration'] if 'min_iteration' in self.args else 50
        
        step_size = self.args['step_size'] if 'step_size' in self.args else 10
           
        optimizer = torch.optim.Adam(classwise_simplex_query+classwise_simplex_refrain, lr=lr)
        #optimizer = torch.optim.SGD(self.classwise_simplex_query+self.classwise_simplex_refrain, lr=lr)

        scheduler_query = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
        
        # 1. Precompute features for query_dataset
        query_dataset_features = self._compute_features(self.query_dataset, embedding_type, layer_name, gradType,True)
        query_dataset_len = len(self.query_dataset)
    
        # 2. Precompute features for unlabeled_dataset
        unlabeled_dataset_features = self._compute_features(self.unlabeled_dataset, embedding_type, layer_name, gradType,False)
        unlabeled_dataset_len = len(self.unlabeled_dataset)
        
        #multiclass selection
        #if self.args has iterations, use that else use 100
        iterations = self.args['iterations'] if 'iterations' in self.args else 100
        
       

        #first get query and refrain params ready
        
        for i in range(iterations):
            # Create lists to store the loss values           
           
            #print('entering iterations')
            # Initialize total loss as a tensor with requires_grad=True
            loss = 0.0
            total_loss=0.0
            optimizer.zero_grad()
            #calculate loss classwise in query dataset
            
            for class_idx in range(num_classes):
                #print('entering classwisecalculation')
                #filter query dataset based on class_idx
                class_mask = torch.stack([item[1] for item in self.query_dataset]) == unique_labels[class_idx]
                #find length of query dataset for that class_idx
                num_query_instances = len(query_dataset_features[torch.nonzero(class_mask).squeeze()])
               
                # For every non-class_idx, select equal instances
                num_samples_per_class = num_query_instances // (num_classes - 1)
                refrain_indices = []
                for other_class_idx in unique_labels:
                    
                    if other_class_idx != class_idx:
                        other_class_mask = torch.stack([item[1] for item in self.query_dataset]) == unique_labels[other_class_idx]
                        other_class_indices = torch.nonzero(other_class_mask).squeeze().tolist()
                        if(len(other_class_indices)>num_samples_per_class):
                            # Randomly sample indices without replacement
                            selected_indices = random.sample(other_class_indices, num_samples_per_class)
                            refrain_indices.extend(selected_indices)
                        else:
                            refrain_indices.extend(other_class_indices)
                 #Extract refrain_features
                query_features = query_dataset_features[torch.nonzero(class_mask).squeeze()]
                query_features=query_features.detach()
            
                refrain_features = query_dataset_features[torch.tensor(refrain_indices)]
                refrain_features=refrain_features.detach()
                query_features=query_features.to(self.device)
                refrain_features=refrain_features.to(self.device)
                #query_imgs=query_imgs.to(self.device)
                beta = torch.ones(len(query_features), requires_grad=False)/len(query_features)
                beta=beta.to(self.device)
                gamma = torch.ones(len(refrain_features), requires_grad=False)/len(refrain_features)
                gamma=gamma.to(self.device)
                #get simplex_query for that class
                simplex_query = classwise_simplex_query[class_idx]
                simplex_query.requires_grad = True
                #get simplex_refrain for that class
                simplex_refrain = classwise_simplex_refrain[class_idx]
                simplex_refrain.requires_grad=True
                #unlabeled_dataloader = DataLoader(dataset=self.unlabeled_dataset, batch_size=minibatch_size, shuffle=False, sampler=sampler)
                loss_avg_query=0.0
                loss_avg_refrain=0.0
                loss_avg_query_refrain=0.0
                #calc num_batches
                num_batches = math.ceil(unlabeled_dataset_len/minibatch_size)
                if num_batches > 1:
                    # Merge a pathologically small remainder batch into the
                    # previous one instead of leaving it standalone - verified
                    # by direct testing that a very small last batch (e.g. size
                    # 1) can degenerate geomloss's Sinkhorn computation
                    # (ValueError: Maximum allowed size exceeded / divide-by-zero
                    # in its epsilon schedule). The existing min(...) end bound
                    # in the loop below then absorbs the remainder automatically
                    # once num_batches is reduced.
                    last_batch_size = unlabeled_dataset_len - (num_batches - 1) * minibatch_size
                    if last_batch_size < minibatch_size / 2:
                        num_batches -= 1
                #batchiwise WD calculation
            
                for batch_idx in range(num_batches):
                    #print('entering batchwise')
                    # Get the features using the pretrained model

                    # Non-overlapping batch bounds (see wassal_multiclass.py
                    # for the same fix and rationale).
                    begindex = batch_idx*minibatch_size
                    endindex = min((batch_idx+1)*minibatch_size, unlabeled_dataset_len)
                #simplex batch query
                    simplex_batch_query = simplex_query[begindex : endindex]
                #should we average or project?
                    if(simplex_batch_query.sum()!=0):
                        simplex_batch_query = simplex_batch_query.clone() / simplex_batch_query.sum()
                    #simplex_batch_query.requires_grad = True
                    simplex_batch_query=simplex_batch_query.to(self.device)
                    simplex_batch_refrain = simplex_refrain[begindex : endindex]
                    #should we average or project?
                    if(simplex_batch_refrain.sum()!=0):
                        simplex_batch_refrain = simplex_batch_refrain.clone() / simplex_batch_refrain.sum()
                    #simplex_batch_refrain.requires_grad = True
                    simplex_batch_refrain=simplex_batch_refrain.to(self.device)
                    #get minibatch unlabeled features
                    unlabeled_features = unlabeled_dataset_features[begindex : endindex]
                    
                    
                    
                    unlabeled_features=unlabeled_features.to(self.device)
                    loss_avg_query=loss_avg_query+(loss_func(simplex_batch_query, unlabeled_features, beta, query_features) / num_batches)
                    
                    loss_avg_refrain=loss_avg_refrain+(loss_func(simplex_batch_refrain, unlabeled_features, gamma, refrain_features) / num_batches)
                    loss_avg_query_refrain=loss_avg_query_refrain+(loss_func(simplex_batch_query, unlabeled_features, simplex_batch_refrain, unlabeled_features) / num_batches)
                    
            
            #once all batches are done, calculate average loss
            
                total_loss=loss_avg_query+loss_avg_refrain-0.3*loss_avg_query_refrain
                #add to the total loss
                loss = loss + total_loss

            #once one iteration is done for class, do backward and step
            loss.backward()
            optimizer.step()
            scheduler_query.step()
            #project to simplex
                           
            with torch.no_grad():
                for class_idx in range(num_classes):
                    classwise_simplex_query[class_idx].data = self._proj_simplex(classwise_simplex_query[class_idx].data)
                    classwise_simplex_refrain[class_idx].data = self._proj_simplex(classwise_simplex_refrain[class_idx].data)
            
            print("Epoch:[", i,"],Avg loss: [{}]".format(loss),end="\r")
                    #break if loss is less than 1 or greater than -1
                    


            if((loss.item()<0.2 and loss.item()>-0.2) and i>min_iteration):
                break

         #now lets select
        #given budget and num_class find out per class budget
        per_class_budget=math.ceil(budget/num_classes)
        #findout the difference between per class budget and budget
        diff=budget-(per_class_budget*num_classes)

       
        output=[]
        selected_indices=[]
        unique_selected_indices=set()

        # 1. Aggregate Maximum Values Across Classes
        max_across_classes = torch.zeros(len(self.unlabeled_dataset), device=self.device)
        #label of the max across classes
        label_of_max_across_classes = torch.zeros(len(self.unlabeled_dataset), dtype=torch.long, device=self.device)
        for class_idx, simplex_query in enumerate(classwise_simplex_query):
            max_across_classes = torch.max(max_across_classes, simplex_query)
            label_of_max_across_classes = torch.where(simplex_query > max_across_classes, torch.full_like(label_of_max_across_classes, class_idx), label_of_max_across_classes)

        torch.cuda.empty_cache()

        #group max_across_classes label_of_max_across_classes into temp_label_to_simplex_query
        temp_label_to_simplex_query = {}
        for i in range(len(max_across_classes)):
            label = int(label_of_max_across_classes[i].item())
            if label not in temp_label_to_simplex_query:
                temp_label_to_simplex_query[label] = torch.zeros(len(self.unlabeled_dataset), device=self.device)
            temp_label_to_simplex_query[label][i] = max_across_classes[i]


        
        for iteridx,(class_idx, simplex_query) in enumerate(temp_label_to_simplex_query.items()):
            
            

            # 2. Select Least Non-Zero Values
            non_zero_indices = torch.nonzero(simplex_query).squeeze()

            #if non_zero_indices is empty, then we need to add random indices
            if len(non_zero_indices)==0:
                non_zero_indices = torch.randperm(len(simplex_query)).to(self.device)
                
                for idx in non_zero_indices:
                    if idx not in unique_selected_indices:
                        unique_selected_indices.add(idx.item())
                        selected_indices.append(idx.item())
                    if len(selected_indices)==per_class_budget:
                        break
                continue
                
            #if descending is true, the values are sorted in descending order
            sorted_values, sorted_indices = torch.sort(simplex_query[non_zero_indices])
            #add unique indices to the selected_indices by checking with unique_selected_indices
            temp_added_indices=0
            for idx in sorted_indices:
                real_idx = non_zero_indices[idx].item()
                if real_idx not in unique_selected_indices:
                    unique_selected_indices.add(real_idx)
                    selected_indices.append(real_idx)
                    temp_added_indices+=1
                if temp_added_indices==per_class_budget:
                    break
            
                        
            # Mask out the values in simplex_query and simplex_refrain tensors
            # corresponding to selected indices
            masked_simplex_query = simplex_query.clone()
            masked_simplex_refrain = classwise_simplex_refrain[iteridx].clone()
            for idx in selected_indices:
                masked_simplex_query[idx] = 0
                masked_simplex_refrain[idx] = 0
                
            # Each tuple contains the masked simplex_query tensor, masked simplex_refrain tensor, and the class_idx
            output.append((masked_simplex_query.detach().cpu(), masked_simplex_refrain.detach().cpu(), class_idx))
        
        torch.cuda.empty_cache()
        
        #if diff is greater than 0, then we need to add the remaining indices to the selected_indices
        #we will add unique indices that are max across
        if diff>0:
            # 1. Aggregate Maximum Values Across Classes
            max_across_classes = torch.zeros(len(self.unlabeled_dataset), device=self.device)
            for classwise_simplex in classwise_simplex_query:
                max_across_classes = torch.max(max_across_classes, classwise_simplex)

            # 2. Select Least Non-Zero Values
            non_zero_indices = torch.nonzero(max_across_classes).squeeze()
            #if descending is true, the values are sorted in descending order
            sorted_values, sorted_indices = torch.sort(max_across_classes[non_zero_indices])
            #add unique indices to the selected_indices by checking with unique_selected_indices
            temp_added_indices=0
            for idx in sorted_indices:
                real_idx = non_zero_indices[idx].item()
                if real_idx not in unique_selected_indices:
                    unique_selected_indices.add(real_idx)
                    selected_indices.append(real_idx)
                    temp_added_indices+=1
                if temp_added_indices==diff:
                    break
        
        return selected_indices, output


    def select(self, budget):
        return self.select_elements_and_soft(budget)
        