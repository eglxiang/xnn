import xnn
import numpy as np
from bokeh.plotting import output_server, figure, push, curdoc, cursession,vplot
import time

class Loop(object):
    def __init__(self,trainer,learndata=[],validdata=[],metrics={},url=None,savefilenamecsv=None,weightdict={},printflag=True,plotmetricmean=True,savemodelnamebase=None):
        """
        An example training loop.  The trainer will learn from learndata, apply metrics to validdata, plot to bokeh server at url if supplied, and save to csv file.

        :param trainer: The trainer to run
        :param learndata:  A list of generator functions that yield batches of data from which the trainer will learn
        :param validdata:  A list of generator functions that yield batches of data to which the metrics will be applied
        :param metrics:  A list of metrics to apply to the validdata.  Each entry is a tuple, the first element of which is the key in the model.predict output from which the metric should be calculated, the second element of which is the metric
        :param url: A string representing the url where the bokeh server is running (e.g. 'http://127.0.0.1:5006'). If None, or if server cannot be connected, no plotting will occur
        :param savefilenamecsv:  The name of a file into which to write metric results at each epoch.
        :param weightdict:  Dictionary of Weighters.  Keys in weightdict are keys into which the weight results will be inserted in the data dictionary.  Weights are calculated for every training batch.
        :param printflag: Whether to print results to stdout after each epoch
        :param plotmetricmean: Whether to print/save/plot mean of all metrics.  Depending on the metrics, this value might not make sense.
        :param savemodelnamebase: Base name for saving metrics.  If not None, models will be saved for the best values of each metric as looping procedes.
        """
        self.trainer = trainer
        self.learndata = self._listify(learndata)
        self.validdata = self._listify(validdata)
        self.metrics = metrics
        self.weightdict = weightdict
        if url is not None:
            self._plot_flag = self._init_plotsession(url)
            self._datasources = None
        else:
            self._plot_flag = False
        if savefilenamecsv is not None:
            self.savefilenamecsv=savefilenamecsv 
            self._savecsv_flag = True
        else:
            self._savecsv_flag = False
        if savemodelnamebase is not None:
            self.savemodelnamebase = savemodelnamebase
            self._savemodel_flag = True
        else:
            self._savemodel_flag = False
        self._print_flag = printflag
        self._plotmetricmean=plotmetricmean
        self._bestmetvals = None
        self._bestatep = None
        self.meandur = None
        self.ep = 0
        

    def __call__(self,niter=1):
        endep = self.ep+niter
        for ep in xrange(self.ep,endep):
            start = time.time()
            self.ep += 1
            # training
            trainerrs = []
            for ld in self.learndata:
                for batch in ld():
                    batch = self._weight(batch)
                    outs = self.trainer.train_step(batch)
                    trainerrs.append(outs[-1])
            trainerr = np.mean(trainerrs)

            # validation
            metvals = []
            for vd in self.validdata:
                vals = []
                for batch in vd():
                    outs = self.trainer.model.predict(batch)
                    for metkey,met in self.metrics:
                        vals.append(met(outs[metkey],batch))
                metvals.append(vals)
            metvals = np.mean(metvals,axis=0).tolist()
            end = time.time()
            dur = end-start
            if self.meandur is None:
                self.meandur = dur
            else:
                self.meandur = 0.3*self.meandur + 0.7*dur

            #update best values
            isbest = self._update_best(ep,trainerr,metvals)

            #print summary to stdout
            if self._print_flag:
                self._print(ep,trainerr,metvals,dur,endep)

            #plot
            if self._plot_flag:
                self._plot(ep,trainerr,metvals)

            #save to csv
            if self._savecsv_flag:
                self._save(ep,trainerr,metvals)

            #save model if best for any metrics
            isbest_nontrainerr = isbest[1:]
            if self._savemodel_flag and any(isbest_nontrainerr):
                self._savemodel(isbest_nontrainerr)

        if self._print_flag:
            print('Finished %d epochs at %s'%(niter,time.strftime('%I:%M:%S %p')))

    def _weight(self,batch):
        for k,w in self.weightdict.iteritems():
            batch[k] = w(batch)
        return batch

    def _savemodel(self,isbest):
        if self._plotmetricmean:
            if isbest[0]:
                fname = self.savemodelnamebase + '_metricmean'
                self.trainer.model.save_model(fname)
            isbest = isbest[1:] 
        for (mk,mt),ib in zip(self.metrics,isbest):
            if ib:
                fname = self.savemodelnamebase + '_' + mk + '_'+ xnn.metrics.metric_names[mt.metric]
                self.trainer.model.save_model(fname)
    
    def _update_best(self,ep,trainerr,metvals):
        isbest = []
        best = []
        bestatep = []
        if self._plotmetricmean:
            vals = [trainerr] + [np.mean(metvals)] + metvals
        else:
            vals = [trainerr]+metvals
        if self._bestmetvals is None:
            self._bestmetvals = vals
            self._bestatep = [ep]*(len(vals))
            isbest = [True]*(len(vals))
        else:
            for v,b,e in zip(vals,self._bestmetvals,self._bestatep):
                if v < b:
                    isbest.append(True)
                    best.append(v)
                    bestatep.append(ep)
                else:
                    isbest.append(False)
                    best.append(b)
                    bestatep.append(e)
            self._bestmetvals = best
            self._bestatep = bestatep
        return isbest

    def _plot(self,ep,trainerr,metvals):
        if self._datasources is None:
            self._make_figures(ep,trainerr,metvals)
        if self._plotmetricmean:
            vals = [trainerr] + [np.mean(metvals)] + metvals
        else:
            vals = [trainerr] + metvals
        for v,ds in zip(vals,self._datasources):
            self._update_datasource(ds,v,ep)

    def _update_datasource(self,ds,v,ep):
        ds.data['x'].append(ep)
        ds.data['y'].append(v)
        cursession().store_objects(ds)

    def _make_figures(self,ep,trainerr,metvals):
        self._datasources = []
        figures = []
        fig = figure(title='Total Training Cost',x_axis_label='Epoch',y_axis_label='Cost')
        fig.line([ep],[trainerr],name='plot')
        ds = fig.select(dict(name='plot'))[0].data_source
        self._datasources.append(ds)
        if self._plotmetricmean:
            figures.append(fig)
            fig = figure(title='Metric Mean',x_axis_label='Epoch',y_axis_label='Mean')
            fig.line([ep],[np.mean(metvals)],name='plot')
            ds = fig.select(dict(name='plot'))[0].data_source
            self._datasources.append(ds)
            figures.append(fig)
        for mv,(mk,m) in zip(metvals,self.metrics):
            name = xnn.metrics.metric_names[m.metric]
            fig = figure(title=mk,x_axis_label='Epoch',y_axis_label=name)
            fig.line([ep],[mv],name='plot')
            ds = fig.select(dict(name='plot'))[0].data_source
            self._datasources.append(ds)
            figures.append(fig)
        allfigs = vplot(*figures)
        push()

    def _print(self,ep,trainerr,metvals,dur,totalep):
        fmt = '{0:45} {1:20}:   {2:0.4f} (best {3:0.4f} at epoch {4:g})'
        print '=========='
        print('Epoch %d / %d -- %0.2f seconds'%(ep,totalep-1,dur))
        print('Expected Finish: %s'%(self._timedone(ep,totalep)))
        print '----------'
        print(fmt.format('Training total cost','',trainerr,self._bestmetvals[0],self._bestatep[0]))
        print '----------'
        beststartid = 1
        if self._plotmetricmean:
            print fmt.format('Overall Mean','',np.mean(metvals),self._bestmetvals[1],self._bestatep[1])
            print '----------'
            beststartid = 2
        for mv,(mk,m),bv,be in zip(metvals,self.metrics,self._bestmetvals[beststartid:],self._bestatep[beststartid:]):
            name = xnn.metrics.metric_names[m.metric]
            print fmt.format(name,mk,mv,bv,be)
            
    def _timedone(self,ep,totalep):
        now = time.time()
        done = now + self.meandur*(totalep-1-ep)
        ts = time.localtime(done)
        return time.strftime('%a, %I:%M:%S %p',ts)

    def _save(self,ep,trainerr,metvals):
        if (not hasattr(self,'_headers_written')) or (not self._headers_written):
            self._save_header()
            self._headers_written = True
        with open(self.savefilenamecsv,'a') as f:
            f.write(('%0.4f,'%trainerr))
            if self._plotmetricmean:
                f.write(('%0.4f,'%np.mean(metvals)))
            for mv in metvals:
                f.write(('%0.4f,'%mv))
            f.write('\n')
            f.flush()
    
    def _save_header(self):
        with open(self.savefilenamecsv,'a') as f:
            f.write('Training cost,')
            if self._plotmetricmean:
                f.write('Overall mean,')
            for mk,m in self.metrics:
                name = xnn.metrics.metric_names[m.metric]
                f.write(name+'_'+mk+',')
            f.write('\n')
            f.flush()

    def _init_plotsession(self,url):
        docname = self.trainer.model.name
        try:
            output_server(docname,url=url)
        except:
            return False
        d = curdoc()
        self.plot_address = '%s/bokeh/doc/%s/%s'%(url,d.docid,d.ref['id'])
        print('plots available at: %s'%(self.plot_address))
        return True



    def _listify(self,thing):
        if not isinstance(thing,list):
            return [thing]
        return thing
