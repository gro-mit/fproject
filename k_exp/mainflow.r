setwd('D:\\Code Files\\fproject\\k_exp')
library(ggplot2)
library(GGally)
library(plot3D)
library(fpc)
library(cluster)
library(apcluster)
require(vegan)
source('kcomp_functions.r')

D = c(5,10,15)
trueK = c(3,5,7)
maxK = 9
res = data.frame()
seed = 721
for(k in trueK)
{
	for(d in D)
	{	
		set.seed(d*k+seed)
		data = make_data(d, k)
		# for 2-D dimensional data
		#scatterplot_data(datax) + coord_fixed() + ggtitle("Original Data")

		# for 3-D or higher dimensional data, you can also plot the 3-D projection
		# plot3d_data(datax)
		data0 = data[, 1:d, drop = FALSE]
		pb_res = find_clusters_scan(data0, maxK)
		cc_res = cascadeKM(data0, 1, maxK, iter = 1000)
		wss = (nrow(data0)-1)*sum(apply(data0,2,var))
		sihouette_score = numeric(maxK)
		for(i in 2:maxK)
		{
			wss[i] = sum(kmeans(data0, centers = i)$withinss)
			sihouette_score[i] = pam(data0, i)$silinfo$avg.width
		}
		pb_k = pb_res$nclusters
		sihouette_k = which.max(sihouette_score)
		sse_df = data.frame(k=1:maxK, sse=wss)
		cc_k = as.numeric(which.max(cc_res$results[2,]))
		tmp_res = c(k, d, 0, sihouette_k, pb_k, cc_k)
		res = rbind(res, tmp_res)
		plt = ggplot(sse_df, aes(x=k,y=sse))+geom_line()+geom_point(size=2,shape=19)+scale_x_continuous(breaks = c(2,4,6,8))+labs(title="Elbow for Kmeans clustering",x="Number of clusters",y="Within groups sum of squares")+theme(plot.title = element_text(hjust = 0.5))
		print(plt)
		png_name = paste(seed,"elbow_trueK_",k,"_dim_",d,".png")
		ggsave(filename = png_name, plot = plt, dpi = 300)
		cat("trueK is", k, "dim is", d, "\n")
		cat("elbow plot number of clusters shown as follow:\n")
		cat("silhouette-optimal number of clusters:", sihouette_k, "\n")
		cat("parametric boostrap number of clusters:", pb_k, "\n")
		cat("calinski criterion number of clusters:", cc_k, "\n")
	}
}
names(res) = c("trueK", "dim", "elbow", "sihouette", "pb", "cc")
f_name = paste(seed, "res_k_exp.csv")
write.csv(res,f_name, row.names = FALSE) 
