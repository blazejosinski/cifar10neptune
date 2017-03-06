import sys
from collections import OrderedDict
from bunch import Bunch
from deepsense import neptune
from neptune_utils import TimeSeries, NeptuneTimeseriesObserver

class NeptuneMain():

    def create_neptune_channels_charts(self):
        channel_names = [
                'test_accuracy',
                'train_accuracy',
                'batch_processing_time',
                'learning_rate',
                'loss',
                'generator_choice',
                ]
        channels = {}

        for channel_name in channel_names:
            channel = self.job.create_channel(
                name=channel_name,
                channel_type=neptune.ChannelType.NUMERIC,
                is_last_value_exposed=True,
                is_history_persisted=True)
            channels[channel_name] = channel

        
        figures_schema = OrderedDict([
            # format is (channel_name, name_on_the_plot)
            ('loss', [
                ('loss', 's'),
            ]),
            ('train accuracy', [
                ('train_accuracy', 'accuracy'),
            ]),
            ('test accuracy', [
                ('test_accuracy', 'accuracy'),
            ]),
            ('batch generator choice', [
                ('generator_choice', 'generator no'),
            ]),
            ('batch processing times', [
                ('batch_processing_time', 'seconds'),
            ]),
            ('learning rate', [
                ('learning_rate', 'learning rate'),
            ]),
        ])


        for chart_name, d in figures_schema.iteritems():
            chart_series = {}
            for channel_name, name_on_plot in d:
                chart_series[name_on_plot] = channels[channel_name]
                print 'add to chart ', name_on_plot, ' channel ', channel_name

            self.job.create_chart(name=chart_name, series=chart_series)

        ts = Bunch()
        for channel_name in channels.iterkeys():
            ts.__setattr__(channel_name, TimeSeries())

        for figure_title, l in figures_schema.iteritems():
            for idx, (channel_name, line_name) in enumerate(l):
                observer = NeptuneTimeseriesObserver(
                    name=channel_name,
                    channel=channels[channel_name], add_freq=1)
                getattr(ts, channel_name).add_add_observer(observer)

        return ts



    def go(self):
        self.ctx = neptune.Context(sys.argv)
        self.job = self.ctx.job
        self.args = self.ctx.params
        self.ts = self.create_neptune_channels_charts()

        sin_channel = self.ctx.job.create_channel(name='sin', channel_type=neptune.ChannelType.NUMERIC)
        sin_channel.send(x=0, y=10)
        sin_channel.send(x=1, y=5)

        from cifar10tf import Cifar_boxes
        self.cifar = Cifar_boxes(self.args, self.ts)

        self.cifar.run()


def main():
    sys.exit(NeptuneMain().go())

if __name__ == '__main__':
    main()
