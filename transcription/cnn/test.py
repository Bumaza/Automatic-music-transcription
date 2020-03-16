from transcription.cnn import alex_net, res_net

an = alex_net.AlexNet()
an.create()
an.plot()

rs = res_net.ResNet()
rs.create()
rs.plot()