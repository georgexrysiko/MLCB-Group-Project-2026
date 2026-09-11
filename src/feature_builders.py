import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.feature_selection import (RFE,SelectKBest,mutual_info_classif,)
from sklearn.linear_model import LogisticRegression
from helper_functions import (aggregate_matrix_by_sample, combine_features,labels_for_samples,to_float32_dense)

SEED = 42
#  ======================================================================
# Base class used by all fold-specific feature builders, where
# each builder is fitted again using only the training data of each LOPO fold.

class FeatureBuilder:
    def fit(self, genes, pathways, y, group_ids):
        raise NotImplementedError

    def transform(self, genes, pathways):
        raise NotImplementedError

    def get_feature_names( self, gene_names,
        pathway_names):
        raise NotImplementedError

# ======================================================================
# Returning all gene-expression features without feature selectionor dimensionality reduction.

class GenesBuilder(FeatureBuilder):
    def fit(self, genes, pathways, y, group_ids):
        # Nothing needs to be fitted for this feature set.
        return self

    def transform(self,genes,pathways,):
        if sp.issparse(genes):
            return genes.astype(np.float32)

        return np.asarray(genes, dtype=np.float32)

    def get_feature_names(self,gene_names,pathway_names):
        return list(gene_names)

# ======================================================================
# Returns only the PROGENy pathway scores.

class PathwaysBuilder(FeatureBuilder):
    def fit(self,genes,pathways,y,group_ids):
        # Nothing needs to be fitted because the pathway scores already exist.
        return self

    def transform(self, genes,pathways):
        return np.asarray(pathways, dtype=np.float32,)

    def get_feature_names(self,gene_names,pathway_names):
        return list(pathway_names)

# ======================================================================
# All genes + all PROGENy pathways

class GenesPathwaysBuilder(FeatureBuilder):

    def fit(self, genes, pathways, y, group_ids):
        # Nothing needs to be fitted for this feature set.
        return self

    def transform(self,genes,pathways):
        return combine_features(genes,pathways)

    def get_feature_names(self,gene_names,pathway_names):
        return (list(gene_names) + list(pathway_names))

# ======================================================================
# PCA and PCA + pathways
# ======================================================================

class PCABuilder(FeatureBuilder):

    def __init__(self, n_components: int,add_pathways: bool):
        self.n_components = int(n_components)
        self.add_pathways = bool(add_pathways)
        self.pca = None

    def fit(self,genes,pathways,y,group_ids):
        # Converting the training gene matrix to dense format for PCA.
        dense_genes = to_float32_dense(genes)

        # PCA cannot use more components than available cells or genes.
        max_components = min(self.n_components,
                             dense_genes.shape[0] - 1,
                             dense_genes.shape[1])

        if max_components < 1:
            raise ValueError("Not enough training cells or genes for PCA.")

        # PCA is fitted only on the training cells of the current fold.
        self.pca = PCA(n_components=max_components,svd_solver="randomized",random_state=SEED)
        self.pca.fit(dense_genes)

        return self

    def transform(self,genes,pathways,):
        if self.pca is None:
            raise RuntimeError("PCABuilder must be fitted before transform().")

        # Applying the PCA fitted on the training cells.
        pcs = (self.pca.transform(to_float32_dense(genes))
               .astype( np.float32,copy=False, ))

        # Adding pathway scores only for the PCA + pathways condition.
        if self.add_pathways:
            return combine_features(pcs,pathways)

        return pcs

    def get_feature_names(self,gene_names,pathway_names):
        if self.pca is None:
            raise RuntimeError("PCABuilder must be fitted before requesting feature names.")

        pc_names = [f"PC{index + 1}" for index in range( self.pca.n_components_)]
        if self.add_pathways:
            return (pc_names + list(pathway_names))
        return pc_names


    def get_explained_variance(self) -> pd.DataFrame:

        if self.pca is None:
            raise RuntimeError("PCABuilder must be fitted before requesting explained variance.")

        variance_ratio = (self.pca.explained_variance_ratio_)
        return pd.DataFrame({"PC": [f"PC{index + 1}" for index in range(len(variance_ratio))],
                             "explained_variance_ratio": variance_ratio,
                             "cumulative_explained_variance": np.cumsum(variance_ratio)}
                             )


# ======================================================================
#  Mutual Information genes and MI genes + pathways
# Selecting top-k genes using sample-level balanced Mutual Information.
# MI is fitted only on biological samples from the current training fold.

class MIBuilder(FeatureBuilder):

    def __init__(self, k: int, add_pathways: bool):
        self.k = int(k)
        self.add_pathways = bool(add_pathways)
        self.selector = None
        self.balanced_sample_counts_ = None

    def fit(self, genes, pathways, y, group_ids):

        if genes.shape[1] == 0:
            raise ValueError("MIBuilder received no gene features.")

        sample_ids_array = np.asarray(group_ids).astype(str)
        y_array = np.asarray(y, dtype=int)

        if len(sample_ids_array) != len(y_array):
            raise ValueError("Sample IDs and y must have the same length inside MIBuilder.")

        if genes.shape[0] != len(y_array):
            raise ValueError("genes and y must have the same number of rows inside MIBuilder.")

        # Aggregate training cells to one row per biological sample.
        sample_genes, unique_samples = aggregate_matrix_by_sample(matrix=genes,
                                                                  biological_sample_ids=sample_ids_array
                                                                  )

        sample_labels = labels_for_samples(biological_sample_ids=sample_ids_array,
                                           labels=y_array,
                                           unique_samples=unique_samples
)

        classes, counts = np.unique(sample_labels, return_counts=True)

        if len(classes) < 2:
            raise ValueError("Mutual Information requires both response classes among the training biological samples.")

        # Deterministically balance the response classes at sample level.
        rng = np.random.default_rng(SEED)
        target_count = int(counts.max())
        balanced_indices = []

        for class_label in classes:
            class_indices = np.flatnonzero(sample_labels == class_label)

            sampled_indices = rng.choice(class_indices,size=target_count,replace=(len(class_indices) < target_count))
            balanced_indices.extend(sampled_indices.tolist())

        balanced_indices = np.asarray( balanced_indices,dtype=int)
        balanced_indices = rng.permutation(balanced_indices)
        balanced_genes = np.asarray(sample_genes[balanced_indices], dtype=np.float32 )
        balanced_labels = np.asarray(sample_labels[balanced_indices],dtype=int)

        self.balanced_sample_counts_ = {int(label): int(count) for label, count in zip( *np.unique(balanced_labels, return_counts=True))}

        k = min(self.k, balanced_genes.shape[1])

        if k < 1:
            raise ValueError( "MIBuilder must select at least one gene.")

        self.selector = SelectKBest( score_func=lambda X, target: mutual_info_classif(
            X,target,discrete_features=False, random_state=SEED ), 
            k=k )

        self.selector.fit( balanced_genes, balanced_labels)

        return self

    def transform(self, genes, pathways):

        if self.selector is None:
            raise RuntimeError("MIBuilder must be fitted before transform().")

        selected_genes = (self.selector.transform(to_float32_dense(genes)).astype(np.float32,copy=False))

        if self.add_pathways:
            return combine_features(selected_genes,pathways)
        return selected_genes

    def get_feature_names(self, gene_names, pathway_names):

        if self.selector is None:
            raise RuntimeError("MIBuilder must be fitted before requesting feature names.")

        selected_gene_names = np.asarray(gene_names)[self.selector.get_support()].tolist()

        if self.add_pathways:
            return selected_gene_names + list(pathway_names)

        return selected_gene_names

# ======================================================================
# RFE-selected PCA components and RFE-PCA + pathways

class RFEPCBuilder(FeatureBuilder):

    def __init__(self, n_pcs: int, n_features: int, add_pathways: bool):
        self.n_pcs = int(n_pcs)
        self.n_features = int(n_features)
        self.add_pathways = bool(add_pathways)
        self.pca_builder = PCABuilder( n_components=self.n_pcs,add_pathways=False)
        self.rfe = None
        self.unique_training_samples_ = None
        self.training_sample_labels_ = None

    def fit(self, genes, pathways, y, group_ids):

        # group_ids contains patient_timepoint
        sample_ids_array = np.asarray(group_ids).astype(str)
        y_array = np.asarray(y, dtype=int)

        if len(sample_ids_array) != len(y_array):
            raise ValueError("Sample IDs and y must have the same length inside RFEPCBuilder.")

        if genes.shape[0] != len(y_array):
            raise ValueError(f"genes contains {genes.shape[0]} rows but y contains {len(y_array)} labels.")

        # Fitting PCA only on the training cells.
        self.pca_builder.fit(genes=genes, pathways=pathways,y=y_array,group_ids=sample_ids_array)

        cell_pcs = self.pca_builder.transform(genes=genes, pathways=pathways)

        # Aggregate PCA coordinates to one row per biological sample.
        sample_pcs, unique_samples = aggregate_matrix_by_sample( matrix=cell_pcs, biological_sample_ids=sample_ids_array )
        sample_labels = labels_for_samples( biological_sample_ids=sample_ids_array, labels=y_array, unique_samples=unique_samples)

        if sample_pcs.shape[0] != len(sample_labels):
            raise ValueError( "The number of aggregated biological samples does not match the number of sample labels.")

        if np.unique(sample_labels).size < 2:
            raise ValueError("RFE requires both response classes among the training biological samples.")

        self.unique_training_samples_ = np.asarray(unique_samples).astype(str)
        self.training_sample_labels_ = np.asarray(sample_labels,dtype=np.int8)
        n_select = min(self.n_features,sample_pcs.shape[1] )

        if n_select < 1:
            raise ValueError("RFE must select at least one PCA component.")

        # RFE is fitted on one row per biological sample.
        estimator = LogisticRegression( penalty="l2",solver="liblinear",class_weight="balanced",random_state=SEED, max_iter=2000)
        self.rfe = RFE(estimator=estimator,n_features_to_select=n_select,step=1)
        self.rfe.fit(sample_pcs, sample_labels)

        return self

    def transform(self, genes, pathways):

        if self.rfe is None:
            raise RuntimeError( "RFEPCBuilder must be fitted before transform().")

        pcs = self.pca_builder.transform( genes=genes,pathways=pathways )
        selected_pcs = (self.rfe.transform(pcs).astype(np.float32,copy=False))

        if self.add_pathways:
            return combine_features(selected_pcs,pathways)

        return selected_pcs

    def get_feature_names(self, gene_names, pathway_names):

        if self.rfe is None:
            raise RuntimeError("RFEPCBuilder must be fitted before requesting feature names.")

        all_pc_names = self.pca_builder.get_feature_names(gene_names=gene_names,
            pathway_names=[])

        selected_pc_names = np.asarray(all_pc_names)[self.rfe.get_support()].tolist()

        if self.add_pathways:
            return selected_pc_names + list(pathway_names)

        return selected_pc_names


    def get_selected_component_ranking(self) -> pd.DataFrame:

        if self.rfe is None:
            raise RuntimeError("RFEPCBuilder must be fitted before requesting the component ranking.")

        pc_names = [ f"PC{index + 1}" for index in range(len(self.rfe.support_))]

        return (pd.DataFrame({"feature": pc_names, 
                              "selected": self.rfe.support_,
                              "ranking": self.rfe.ranking_})
                              .sort_values(["selected", "ranking"], ascending=[False, True]).reset_index(drop=True))


    def get_training_sample_summary(self) -> pd.DataFrame:
        if (self.unique_training_samples_ is None or self.training_sample_labels_ is None):
            raise RuntimeError("RFEPCBuilder must be fitted before requesting its training-sample summary.")

        return pd.DataFrame({ "sample_id": self.unique_training_samples_,
                             "true_label": self.training_sample_labels_})