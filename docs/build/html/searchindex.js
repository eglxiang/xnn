Search.setIndex({envversion:46,filenames:["index","modules/modules","modules/xnn","modules/xnn.adjusters","modules/xnn.data","modules/xnn.experiments","modules/xnn.init","modules/xnn.layers","modules/xnn.metrics","modules/xnn.model","modules/xnn.nonlinearities","modules/xnn.objectives","modules/xnn.training","modules/xnn.utils"],objects:{"":{xnn:[2,0,0,"-"]},"xnn.data":{HDF5BatchLoad:[4,0,0,"-"],HDF5FieldReader:[4,0,0,"-"],HDF5Preprocessors:[4,0,0,"-"],HDF5RamPool:[4,0,0,"-"],samplers:[4,0,0,"-"],weights:[4,0,0,"-"]},"xnn.data.HDF5BatchLoad":{HDF5BatchLoad:[4,2,1,""],main:[4,3,1,""]},"xnn.data.HDF5BatchLoad.HDF5BatchLoad":{addReaders:[4,1,1,""],datakeys:[4,1,1,""],num_batches:[4,1,1,""],to_dict:[4,1,1,""]},"xnn.data.HDF5FieldReader":{HDF5FieldReader:[4,2,1,""]},"xnn.data.HDF5FieldReader.HDF5FieldReader":{getName:[4,1,1,""],readBatch:[4,1,1,""],to_dict:[4,1,1,""]},"xnn.data.HDF5Preprocessors":{ageG_to_hard:[4,2,1,""],ageG_to_soft:[4,2,1,""],makeFloatX:[4,3,1,""],pixelPreprocess:[4,2,1,""]},"xnn.data.HDF5Preprocessors.ageG_to_hard":{to_dict:[4,1,1,""]},"xnn.data.HDF5Preprocessors.ageG_to_soft":{to_dict:[4,1,1,""]},"xnn.data.HDF5Preprocessors.pixelPreprocess":{to_dict:[4,1,1,""]},"xnn.data.HDF5RamPool":{HDF5RamPool:[4,2,1,""],PoolMerger:[4,2,1,""]},"xnn.data.HDF5RamPool.HDF5RamPool":{datakeys:[4,1,1,""],does_pool_update:[4,1,1,""],nInPool:[4,1,1,""],to_dict:[4,1,1,""]},"xnn.data.samplers":{BinarySampler:[4,2,1,""],BinnedSampler:[4,2,1,""],CategoricalSampler:[4,2,1,""],Sampler:[4,2,1,""]},"xnn.data.samplers.BinarySampler":{add_other_sample:[4,1,1,""],reset_batch:[4,1,1,""],to_dict:[4,1,1,""]},"xnn.data.samplers.BinnedSampler":{add_other_sample:[4,1,1,""],reset_batch:[4,1,1,""],to_dict:[4,1,1,""]},"xnn.data.samplers.CategoricalSampler":{add_other_sample:[4,1,1,""],reset_batch:[4,1,1,""],to_dict:[4,1,1,""]},"xnn.data.samplers.Sampler":{to_dict:[4,1,1,""]},"xnn.data.weights":{BinaryWeighter:[4,2,1,""],BinnedWeighter:[4,2,1,""],CategoricalWeighter:[4,2,1,""],Weighter:[4,2,1,""]},"xnn.data.weights.BinaryWeighter":{to_dict:[4,1,1,""]},"xnn.data.weights.BinnedWeighter":{to_dict:[4,1,1,""]},"xnn.data.weights.CategoricalWeighter":{to_dict:[4,1,1,""]},"xnn.data.weights.Weighter":{to_dict:[4,1,1,""]},"xnn.experiments":{experiment:[5,0,0,"-"]},"xnn.experiments.experiment":{Experiment:[5,2,1,""],ExperimentCondition:[5,2,1,""]},"xnn.experiments.experiment.Experiment":{add_factor:[5,1,1,""],get_all_conditions_changes:[5,1,1,""],get_conditions_iterator:[5,1,1,""],get_conditions_slice_iterator:[5,1,1,""],get_nth_condition:[5,1,1,""],get_nth_condition_changes:[5,1,1,""],get_num_conditions:[5,1,1,""],to_dict:[5,1,1,""]},"xnn.experiments.experiment.ExperimentCondition":{to_dict:[5,1,1,""]},"xnn.init":{init:[6,0,0,"-"]},"xnn.init.init":{MaskedInit:[6,2,1,""]},"xnn.init.init.MaskedInit":{sample:[6,1,1,""]},"xnn.layers":{local:[7,0,0,"-"],noise:[7,0,0,"-"],normalization:[7,0,0,"-"],prelu:[7,0,0,"-"],transpose:[7,0,0,"-"]},"xnn.layers.local":{LocalLayer:[7,2,1,""]},"xnn.layers.local.LocalLayer":{get_output_for:[7,1,1,""],get_output_shape_for:[7,1,1,""]},"xnn.layers.noise":{GaussianDropoutLayer:[7,2,1,""]},"xnn.layers.noise.GaussianDropoutLayer":{get_output_for:[7,1,1,""]},"xnn.layers.normalization":{BatchNormLayer:[7,2,1,""],ContrastNormLayer:[7,2,1,""]},"xnn.layers.normalization.BatchNormLayer":{get_output_for:[7,1,1,""],get_output_shape_for:[7,1,1,""]},"xnn.layers.normalization.ContrastNormLayer":{get_output_for:[7,1,1,""]},"xnn.layers.prelu":{PReLULayer:[7,2,1,""]},"xnn.layers.prelu.PReLULayer":{get_output_for:[7,1,1,""]},"xnn.layers.transpose":{TransposeDenseLayer:[7,2,1,""],TransposeLocalLayer:[7,2,1,""]},"xnn.layers.transpose.TransposeDenseLayer":{get_output_for:[7,1,1,""]},"xnn.layers.transpose.TransposeLocalLayer":{get_output_for:[7,1,1,""]},"xnn.metrics":{metricsSuite:[8,0,0,"-"]},"xnn.metrics.metricsSuite":{Metric:[8,2,1,""],baseline:[8,3,1,""],compute2AFC:[8,3,1,""],computeBalancedErrorRate:[8,3,1,""],computeBalancedExponentialCost:[8,3,1,""],computeBalancedLogisticAndExponentialCosts:[8,3,1,""],computeBalancedLogisticCost:[8,3,1,""],computeBinarizedBalancedErrorRateBinary:[8,3,1,""],computeBinarizedBalancedErrorRateCategorical:[8,3,1,""],computeBinarizedBalancedExponentialCost:[8,3,1,""],computeBinarizedBalancedLogisticCost:[8,3,1,""],computeBinarizedF1:[8,3,1,""],computeBinarizedHitRate:[8,3,1,""],computeBinarizedJunkRate:[8,3,1,""],computeBinarizedSpecificity:[8,3,1,""],computeCategoricalCrossentropy:[8,3,1,""],computeCondProb:[8,3,1,""],computeConfusionMatrix:[8,3,1,""],computeEqualErrorRate:[8,3,1,""],computeErrorRateDiffSquared:[8,3,1,""],computeF1:[8,3,1,""],computeF:[8,3,1,""],computeHitRate:[8,3,1,""],computeJCorr:[8,3,1,""],computeJunkRate:[8,3,1,""],computeKLDivergence:[8,3,1,""],computeOneHotAccuracy:[8,3,1,""],computeOptimalBalancedErrorRate:[8,3,1,""],computeOptimalBalancedErrorRateCategorical:[8,3,1,""],computeOptimalBalancedExponentialCost:[8,3,1,""],computeOptimalBalancedLogisticCost:[8,3,1,""],computeOptimalF1:[8,3,1,""],computePercentCorrect:[8,3,1,""],computePrecision:[8,3,1,""],computeSpecificity:[8,3,1,""],computeThresholdPercentCorrect:[8,3,1,""],confMatAggregate:[8,3,1,""],convertLLRtoProb:[8,3,1,""],optimizeOverBaselinesAndScales:[8,3,1,""],optimizeOverThresholds:[8,3,1,""],threshold:[8,3,1,""]},"xnn.metrics.metricsSuite.Metric":{to_dict:[8,1,1,""]},"xnn.model":{model:[9,0,0,"-"]},"xnn.model.model":{Model:[9,2,1,""]},"xnn.model.model.Model":{add_full_net_from_layer:[9,1,1,""],add_layer:[9,1,1,""],bind_eval_output:[9,1,1,""],bind_input:[9,1,1,""],bind_output:[9,1,1,""],from_dict:[9,1,1,""],from_dict_static:[9,5,1,""],load_model:[9,1,1,""],make_bound_input_layer:[9,1,1,""],make_dense_drop_stack:[9,1,1,""],make_dense_layer:[9,1,1,""],make_dropout_layer:[9,1,1,""],predict:[9,1,1,""],save_model:[9,1,1,""],to_dict:[9,1,1,""]},"xnn.nonlinearities":{nonlinearities:[10,0,0,"-"]},"xnn.nonlinearities.nonlinearities":{hard_sigmoid:[10,3,1,""],scale:[10,2,1,""],sigmoid_evidence:[10,3,1,""],softmax_evidence:[10,3,1,""]},"xnn.objectives":{objectives:[11,0,0,"-"]},"xnn.objectives.objectives":{absolute_error:[11,3,1,""],binary_crossentropy:[11,3,1,""],categorical_crossentropy:[11,3,1,""],cross_covariance:[11,2,1,""],from_dict:[11,3,1,""],hinge_loss:[11,2,1,""],kl_divergence:[11,3,1,""],squared_hinge_loss:[11,2,1,""]},"xnn.objectives.objectives.cross_covariance":{from_dict:[11,1,1,""],to_dict:[11,1,1,""]},"xnn.objectives.objectives.hinge_loss":{from_dict:[11,1,1,""],to_dict:[11,1,1,""]},"xnn.objectives.objectives.squared_hinge_loss":{from_dict:[11,1,1,""],to_dict:[11,1,1,""]},"xnn.training":{loop:[12,0,0,"-"],trainer:[12,0,0,"-"]},"xnn.training.loop":{Loop:[12,2,1,""]},"xnn.training.trainer":{ParamUpdateSettings:[12,2,1,""],Trainer:[12,2,1,""]},"xnn.training.trainer.ParamUpdateSettings":{to_dict:[12,1,1,""]},"xnn.training.trainer.Trainer":{bind_global_update:[12,1,1,""],bind_regularization:[12,1,1,""],bind_update:[12,1,1,""],get_cost:[12,1,1,""],get_outputs:[12,1,1,""],get_update:[12,1,1,""],init_ins_variables:[12,1,1,""],set_model:[12,1,1,""],to_dict:[12,1,1,""],train_step:[12,1,1,""]},"xnn.utils":{utils:[13,0,0,"-"]},"xnn.utils.utils":{GracefulStop:[13,2,1,""],Progbar:[13,2,1,""],Tnanmax:[13,3,1,""],Tnanmean:[13,3,1,""],Tnansum:[13,3,1,""],draw_to_file:[13,3,1,""],lengthExpection:[13,4,1,""],noProbVectorException:[13,4,1,""],nonNegativeExpection:[13,4,1,""],numpy_one_hot:[13,3,1,""],probEmbarrasingMistakeForAge:[13,3,1,""],theano_digitize:[13,3,1,""],typechecker:[13,3,1,""]},"xnn.utils.utils.GracefulStop":{handler:[13,1,1,""]},"xnn.utils.utils.Progbar":{add:[13,1,1,""],update:[13,1,1,""]}},objnames:{"0":["py","module","Python module"],"1":["py","method","Python method"],"2":["py","class","Python class"],"3":["py","function","Python function"],"4":["py","exception","Python exception"],"5":["py","staticmethod","Python static method"]},objtypes:{"0":"py:module","1":"py:method","2":"py:class","3":"py:function","4":"py:exception","5":"py:staticmethod"},terms:{"_get_pydot_graph":13,"class":[4,5,6,7,8,9,10,11,12,13],"default":5,"final":9,"float":[9,12],"function":[7,9,10,12],"import":9,"int":[4,5,9],"return":[4,5,9],"static":9,"true":[4,7,9,12],absolut:11,absolute_error:11,across:9,activ:10,adapt:8,add:[4,5,9,13],add_factor:5,add_full_net_from_lay:9,add_lay:9,add_other_sampl:4,addread:4,adjust:[],advantag:9,after:12,again:[],ageg_to_hard:4,ageg_to_soft:4,aggreg:9,aggregation_typ:[8,9],all:[5,9,12],all_outs_dict:12,alpha:[7,8],also:9,although:9,ani:[],anneal:[],anneal_coef:[],anoth:9,appli:[9,10,12],apply_penalti:12,appropri:5,approx:10,approxim:10,arg:[7,11],argument:[],arrai:9,associ:9,assum:10,attach:12,attr:[],attribut:7,autofunct:[],automat:9,averag:[10,13],axi:13,bar:13,base:[4,5,6,7,8,9,11,12],baselin:8,batch:[4,7,9,12],batch_dict:12,batchnormlay:7,batchread:4,batchsiz:4,best:12,beta:8,between:[5,11],bin:[4,13],binary_crossentropi:11,binarysampl:4,binaryweight:4,bind:9,bind_eval_output:9,bind_global_upd:12,bind_input:9,bind_output:9,bind_regular:12,bind_upd:12,binnedsampl:4,binnedweight:4,bokeh:12,bold:[],bool:9,both:9,bound:9,broken:[],build:9,bullet:[],calcul:12,callabl:[],can:[],cannot:12,cartesian:5,categorical_crossentropi:11,categoricalsampl:4,categoricalweight:4,chang:[],changepoint:[],chapter:[],clipsdata:8,code:[],coef:7,coeff:12,coeffic:12,coeffici:12,collect:[8,9],compar:10,compat:12,comput:[9,11],compute2afc:8,computebalancederrorr:8,computebalancedexponentialcost:8,computebalancedlogisticandexponentialcost:8,computebalancedlogisticcost:8,computebinarizedbalancederrorratebinari:8,computebinarizedbalancederrorratecategor:8,computebinarizedbalancedexponentialcost:8,computebinarizedbalancedlogisticcost:8,computebinarizedf1:8,computebinarizedhitr:8,computebinarizedjunkr:8,computebinarizedspecif:8,computecategoricalcrossentropi:8,computecondprob:8,computeconfusionmatrix:8,computeequalerrorr:8,computeerrorratediffsquar:8,computef1:8,computef:8,computehitr:8,computejcorr:8,computejunkr:8,computekldiverg:8,computeonehotaccuraci:8,computeoptimalbalancederrorr:8,computeoptimalbalancederrorratecategor:8,computeoptimalbalancedexponentialcost:8,computeoptimalbalancedlogisticcost:8,computeoptimalf1:8,computepercentcorrect:8,computeprecis:8,computespecif:8,computethresholdpercentcorrect:8,condit:5,confmataggreg:8,connect:[9,12],consid:9,consist:[],constant:[7,9],construct:9,contain:[4,5,9],content:[],continu:5,contrast:7,contrastnormlay:7,conveni:9,convertllrtoprob:8,correspond:10,cost:9,costtot:12,countoth:4,creat:9,cross:5,cross_covari:11,csv:12,current:[9,13],data:[],data_dict:9,data_in_gpu:9,datakei:4,dataset:4,datasharedvardict:12,default_condit:5,defin:5,definit:[],dens:[7,9],denselay:7,depend:[9,12],descript:[],design:5,detector:8,determinist:7,diagram:13,dict:[4,5,9],dictionari:[4,5,9,12],differ:[5,9,11],digit:13,direct:12,displai:13,docstr:13,doe:[7,10],does_pool_upd:4,draw:13,draw_to_fil:13,drop_p_list:9,drop_typ:9,drop_type_list:9,dropout:9,drouput:7,each:[9,12],edgeprotect:7,edu:7,either:[9,12],element:[11,12],end:5,entri:12,epoch:12,equal:10,equival:13,eta:7,evalu:[8,9],even:[],everi:12,evid:10,exactli:[],exampl:[9,12],except:13,expect:9,experi:[],experiment:5,experimentcondit:5,exponenti:[],express:[11,12],factor:5,fals:[4,7,9,12,13],faster:10,fed:9,fhuman:13,field:[4,9],file:[4,9,12,13],filenam:[9,13],filepath:4,first:[9,12],fixed_dict:5,flatten:4,float32:10,fname:9,follow:9,frame:13,from:[4,5,9,12,13],from_dict:[9,11],from_dict_stat:9,fulli:9,func:[],gamma:11,gauss:9,gaussian:[7,9],gaussiandropoutlay:7,gener:[5,9,12],get:4,get_all_conditions_chang:5,get_all_lay:13,get_bias_param:[],get_conditions_iter:5,get_conditions_slice_iter:5,get_cost:12,get_nth_condit:5,get_nth_condition_chang:5,get_num_condit:5,get_output:12,get_output_for:7,get_output_shape_for:7,get_param:[],get_upd:12,getnam:4,given:9,global_update_set:12,glorotuniform:7,good:9,gpu:9,gracefulstop:13,graph:9,group:[4,11],handler:13,hard_sigmoid:10,hdf5:4,hdf5batchload:[],hdf5fieldread:[],hdf5preprocessor:[],hdf5rampool:[],hello:[],here:[],hinge_loss:11,hold:[5,9],how:9,http:[7,12],idea:9,imag:7,img_shap:7,implement:7,incom:7,indent:[],independ:10,index:[0,5,9,13],indict:9,inform:5,inherit:[],init:[],init_ins_vari:12,initi:6,input:[7,9,10],input_kei:9,input_lay:9,input_shap:7,input_var:9,inputlabelkei:9,inputread:4,insert:12,insid:4,instanc:5,integ:[5,9],intend:5,intern:9,interv:[],is_eval_output:9,ital:[],item:[5,11],iter:5,itertool:5,jake:8,josh:8,json:5,just:[],keep:9,keepdim:13,kei:[4,5,9,12],keysampl:4,keyword:[],kl_diverg:11,kwarg:[7,8,11,12,13],label:9,labelkei:4,labelslist:4,lasagn:[6,7,9,12],lasang:13,layer:[],layer_dict:12,layer_nam:[9,12],layerlist:12,learn:12,learn_pivot:7,learn_transform:7,learndata:12,length:9,lengthexpect:13,level:5,like:[],likelihood:10,line:[],linear:[7,10],list:[4,5,9,12,13],lname:12,lnamelist:12,load:9,load_model:9,loader:4,local:[],local_filt:7,locallay:7,localmask:7,log10:10,log:10,loop:[],loss_funct:9,main:4,make:[10,12],make_bound_input_lay:9,make_dense_drop_stack:9,make_dense_lay:9,make_dropout_lay:9,make_grayscal:4,makefloatx:4,manag:5,mask:[6,9],maskedinit:6,match:5,math:11,max:12,mean:[5,8,9,12],mean_var:7,member:5,method:[4,5,6,7,8,9,10,11,12,13],metric:[],metricfn:8,metricsdict:[],metricssuit:[],might:12,min:[11,12],missingvalu:4,mode:[7,11],model:[],modul:[],multi:9,multipl:7,must:[5,9],name:[4,5,7,9,12,13],namebas:9,nanmean:9,nanoth:4,nansum:9,nanweighted_mean:9,nanweighted_sum:9,nativ:5,nbatchinpool:4,ndarrai:4,nest:[],net:5,network:[9,13],neural:5,neuron:[9,10],next:[],ninpool:4,nois:[],none:[4,5,7,8,9,11,12,13],nonlin_list:9,nonlinear:[],nonnegativeexpect:13,noprobvectorexcept:13,norm_typ:7,normal:[],num_batch:4,num_unit:[7,9],num_units_list:9,numbatch:4,number:[4,5,9],numclass:13,numpi:[4,9,13],numpy_one_hot:13,numthreshold:8,objdict:11,object:[],obtain:13,occur:12,onli:9,optimizeoverbaselinesandscal:8,optimizeoverthreshold:8,option:[5,13],order:9,other:[],outkei:8,outlay:9,output:[9,10,12,13],output_lay:9,over:[5,9],overwrit:12,packag:[],page:0,paper:7,paragraph:[],param:13,paramet:[4,5,9,10,11,12,13],parametr:5,paramupdateset:12,parent:[],parent_lay:9,parentlay:9,part:10,particular:[5,9],partit:4,pass:9,pdf:7,penalti:12,per:7,percent:9,picklowestfrequ:4,piecewis:10,pivot:7,pixelpreprocess:4,plot:12,plotmetricmean:12,pmachin:13,pooler:4,poolmerg:4,poolsizetoreturn:4,predict:[9,12],prelu:[],prelulay:7,prepend:9,preprocess:4,preprocessfunc:4,print:12,printflag:12,prior:[7,10],probabl:10,probembarrasingmistakeforag:13,proced:12,process:[],product:5,progbar:13,progress:13,provid:9,python:5,queri:5,ratio:10,read:4,readbatch:4,recon:9,reconstruct:9,rectifi:[7,9],refer:[],refreshpoolprop:4,regular:12,remov:10,report:5,repres:[5,9,12],represent:[4,9],requir:7,rescal:7,reset_batch:4,respons:4,result:12,retriev:4,roi:4,rsalakhu:7,run:[9,12],safe:5,same:9,sampl:6,sampleid:4,samplemethod:4,sampler:[],save:[9,12,13],save_model:9,savefilenamecsv:12,savemodelnamebas:12,scalar:[],scale:[9,10],schedul:[],search:0,second:12,see:[7,13],seed:7,sens:12,sequenc:9,sequenti:4,serial:[4,9],serializ:5,server:12,set:5,set_model:12,shape:[6,9],share:9,shift:10,should:[9,12],shouldmaxim:8,show:[],sigma:7,sigmoid:10,sigmoid_evid:10,signal:13,singl:[4,9],slope:10,softmax:10,softmax_evid:10,some:[],sourc:[4,5,6,7,8,9,10,11,12,13],span:[],specif:5,specifi:[5,9,12],squar:7,squared_hinge_loss:11,srivastava14a:7,stack:9,standard:9,start:5,statpool:4,stdout:12,step:13,stop:5,store:9,str:[4,9],string:[5,9,12,13],strong:[],structur:[5,9],subitem:[],submodul:[],subpackag:[],subscript:[],sum:[9,10],superscript:[],suppli:12,susskind:8,symbol:9,target:[9,13],target_typ:9,targkei:8,tensor:11,tensor_or_tensor:[],term:[],test:7,test_absolute_error:[],test_add_invalid_factor_nam:[],test_aggreg:[],test_batch_norm:[],test_binarizedb:[],test_bind_global_upd:[],test_build_model:[],test_cce_and_kl:[],test_compare_to_lasagn:[],test_confmat:[],test_contrast_norm:[],test_convenience_build:[],test_cross_covari:[],test_experi:[],test_experiment_seri:[],test_gaussian_dropout:[],test_hard_sigmoid:[],test_hinge_loss:[],test_kl_diverg:[],test_lay:[],test_loc:[],test_logistic_regression_train:[],test_metr:[],test_model:[],test_nonlinear:[],test_object:[],test_optimized_threshold:[],test_prelu:[],test_regression_metr:[],test_regular:[],test_scal:[],test_seri:[],test_sigmoid_evid:[],test_softmax_evid:[],test_squared_hinge_loss:[],test_train:[],test_trained_model_seri:[],test_trainer_seri:[],test_util:[],text:[],theano:[9,11,12],theano_digit:13,thi:[4,5,9,10,12],thing:9,third:12,those:5,threshold:[8,11],through:[5,9],time:7,titl:[],tnanmax:13,tnanmean:13,tnansum:13,to_dict:[4,5,8,9,11,12],too:[],topolog:9,toronto:7,total:[],track:9,train:[],train_step:12,trainer:[],trainerset:[],transpos:[],transposedenselay:7,transposelocallay:7,treat:9,tupl:[9,12,13],two:11,type:[5,9],typecheck:13,undoc:[],unit:9,unless:9,updat:[12,13],update_set:12,url:12,util:[],validdata:12,valu:[5,9,12,13],value_for_last_step:13,vari:5,variabl:[5,9],variable_kei:5,varianc:9,verbos:13,wai:[],weight:[],weight_kei:9,weightdict:12,weighted_mean:9,weighted_sum:9,weighter:[4,12],weightkei:8,were:9,when:9,where:[5,12],whether:[9,12],which:[4,5,9,12],whitehil:8,whose:[5,9],width:13,wise:11,word:[],work:9,write:12,written:8,www:7,yield:12},titles:["Welcome to xnn&#8217;s documentation!","xnn","xnn package","Adjusters package","Data package","Experiments package","Init package","Layers package","Metrics package","Model package","Nonlinearities package","Objectives package","Training package","Utils package"],titleterms:{adjust:3,content:2,data:4,document:0,experi:5,hdf5batchload:4,hdf5fieldread:4,hdf5preprocessor:4,hdf5rampool:4,indic:0,init:6,layer:7,local:7,loop:12,metric:8,metricssuit:8,model:9,modul:[0,2,3,4,5,6,7,8,9,10,11,12,13],nois:7,nonlinear:10,normal:7,object:11,other:[],packag:[2,3,4,5,6,7,8,9,10,11,12,13],prelu:7,sampler:4,stuff:[],submodul:[],subpackag:2,tabl:0,test_compare_to_lasagn:[],test_experi:[],test_lay:[],test_metr:[],test_model:[],test_nonlinear:[],test_object:[],test_train:[],test_util:[],train:12,trainer:12,transpos:7,user:[],util:13,weight:4,welcom:0,xnn5:[],xnn:[0,1,2,3,4,5,6,7,8,9,10,11,12,13]}})