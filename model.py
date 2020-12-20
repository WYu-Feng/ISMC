from ops import *
from utils import *
from glob import glob
import time
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
import numpy as np

class UGATIT(object) :
    def __init__(self, sess, args):
        self.light = args.light

        if self.light :
            self.model_name = 'UGATIT_light'
        else :
            self.model_name = 'UGATIT'

        self.sess = sess
        self.phase = args.phase
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.dataset_name = args.dataset
        self.augment_flag = args.augment_flag

        self.epoch = args.epoch
        self.iteration = args.iteration
        self.decay_flag = args.decay_flag
        self.decay_epoch = args.decay_epoch

        self.gan_type = args.gan_type

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.init_lr = args.lr
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight
        self.ld = args.GP_ld
        self.smoothing = args.smoothing

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis
        self.n_critic = args.n_critic
        self.sn = args.sn

        self.img_size = args.img_size
        self.img_ch = args.img_ch


        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        self.trainA_dataset = glob('./new_a/'+'*.jpg')
        self.trainB_dataset = glob('./new_b/'+'*.jpg')
        print(self.trainA_dataset)
        #self.dataset_num = max(len(self.trainA_dataset), len(self.trainB_dataset))

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# gan type : ", self.gan_type)
        print("# dataset : ", self.dataset_name)
        #print("# max dataset number : ", self.dataset_num)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# iteration per epoch : ", self.iteration)
        print("# smoothing : ", self.smoothing)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)
        print("# the number of critic : ", self.n_critic)
        print("# spectral normalization : ", self.sn)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)

    ##################################################################################
    # Generator
    ##################################################################################

    def generator(self, x_init, reuse=False, scope="generator"):
        channel = self.ch
        #将x下采样 再 上采样
        with tf.variable_scope(scope, reuse=reuse) :
            #由四个残余块和两个上采样卷积层组成，步长大小为1
            #一、编码（得到内容）
            x = conv(x_init, channel, kernel=7, stride=1, pad=3, pad_type='reflect', scope='conv')
            
            #填充，卷积
            #x=(32, 64, 64, 64)
            x = instance_norm(x, scope='ins_norm')
            x = relu(x)

            # Down-Sampling
            for i in range(2) :
                #两次下采样
                #x1=(32, 32, 32, 128)
                #x2=(32, 16, 16, 256)
                x = conv(x, channel*2, kernel=3, stride=2, pad=1, pad_type='reflect', scope='conv_'+str(i))
                x = instance_norm(x, scope='ins_norm_'+str(i))
                x = relu(x)
                channel = channel * 2

            # Down-Sampling Bottleneck
            for i in range(self.n_res):
                #四次残块
                #增加了特征图的内容表达能力？？？
                #channel=512
                x = resblock(x, channel, scope='resblock_' + str(i))
                #resblock：参数可学习
                    #x1=卷积 激活
                    #x2=#卷积
                    #x=x1+x2

            # Class Activation Map
            #将x中建两维求均值，得到一个全连接
            #将x中建两维求最大，得到一个全连接
            #三、辅助分类器，使用全局平均合并和全局最大合并
                #cam_x_weight用来学习第k个特征图的重要性权重，并引导解码器计算一组域特定的注意特征图、
                #此时相当于只有一种类型
            cam_x = global_avg_pooling(x)
            #对中间的两维度（图片内容维度）求均值
            cam_gap_logit, cam_x_weight = fully_connected_with_w(cam_x, scope='CAM_logit')
            #新建数据
            #全连接层
            #cam_gap_logit=32*1
            x_gap = tf.multiply(x, cam_x_weight)
            #x_gap=(32, 16, 16, 256)
            
            cam_x = global_max_pooling(x)
            #cam_x=(32, 256)
            cam_gmp_logit, cam_x_weight = fully_connected_with_w(cam_x, reuse=True, scope='CAM_logit')
            #不跟新参数，只是计算
            #cam_gmp_logit=(32, 1)
            x_gmp = tf.multiply(x, cam_x_weight)
            #x_gmp=32*1

            cam_logit = tf.concat([cam_gap_logit, cam_gmp_logit], axis=-1)
            #将两次得到的全连接结果连接 (32, 2)
            x = tf.concat([x_gap, x_gmp], axis=-1)
            
            #x=(32, 16, 16, 512)
            #两次的（权重加偏执）与x的乘积连接 (32, 2)
            
            x = conv(x, channel, kernel=1, stride=1, scope='conv_1x1')
            #对x的通道数降为（512降为256）
            x = relu(x)
 
            #最后一维求和，去掉0？？？
            heatmap = tf.squeeze(tf.reduce_sum(x, axis=-1))
            #x=(32, 16, 16, 256)
            #heatmap=(32, 16, 16)
            
            # Gamma, Beta block
            gamma, beta = self.MLP(x, reuse=reuse)
            #更新参数
            #在归一化时，需要的缩放和平移
            #gamma=beta=(32, 256)

            # Up-Sampling Bottleneck
            #二、解码（在内容上添加风格）
            #上采样，放大图片
            for i in range(self.n_res):
                #四次残块
                    #将残差块与AdaLIN装配在一起
                #x1=(32, 16, 16, 256)
                #channel=512
                x = adaptive_ins_layer_resblock(x, channel, gamma, beta, smoothing=self.smoothing, scope='adaptive_resblock' + str(i))

            # Up-Sampling
            for i in range(2) :
                #up_sample+conv 可以看做是反卷积 （deconv）
                x = up_sample(x, scale_factor=2)
                #原图片2倍放大
                
                # 0x_up: (32, 32, 32, 256)
                # 0x_conv: (32, 32, 32, 128)
                
                # 1x_up: (32, 64, 64, 128)
                # 1x_conv: (32, 64, 64, 64)
                x = conv(x, channel//2, kernel=3, stride=1, pad=1, pad_type='reflect', scope='up_conv_'+str(i))
                x = layer_instance_norm(x, scope='layer_ins_norm_'+str(i))
                x = relu(x)

                channel = channel // 2
            #通过两次放大和卷积，得到(32, 64, 64, 64)的矩阵

            x = conv(x, channels=3, kernel=7, stride=1, pad=3, pad_type='reflect', scope='G_logit')
            #输出(32, 64, 64, 3)的矩阵
            #tanh激活
            x = tanh(x)

            return x, cam_logit, heatmap

    def MLP(self, x, use_bias=True, reuse=False, scope='MLP'):
        channel = self.ch * self.n_res
        #4*64 =x两次下采样后的最后通道数
        #x=(32, 16, 16, 256)
        if self.light :
            x = global_avg_pooling(x)

        with tf.variable_scope(scope, reuse=reuse):
            for i in range(2) :
                x = fully_connected(x, channel, use_bias, scope='linear_' + str(i))
                #x1=(32, 256)
                #x2=(32, 256)
                x = relu(x)


            gamma = fully_connected(x, channel, use_bias, scope='gamma')
            beta = fully_connected(x, channel, use_bias, scope='beta')

            gamma = tf.reshape(gamma, shape=[self.batch_size, 1, 1, channel])
            beta = tf.reshape(beta, shape=[self.batch_size, 1, 1, channel])
            #此时gamma==beta？？？
            return gamma, beta

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x_init, reuse=False, scope="discriminator"):
        D_logit = []
        D_CAM_logit = []
        with tf.variable_scope(scope, reuse=reuse) :
            local_x, local_cam, local_heatmap = self.discriminator_local(x_init, reuse=reuse, scope='local')
            global_x, global_cam, global_heatmap = self.discriminator_global(x_init, reuse=reuse, scope='global')
            #_x=(32, 8, 8, 1) ，_cam=(32, 2)
            
            D_logit.extend([local_x, global_x])
            D_CAM_logit.extend([local_cam, global_cam])

            return D_logit, D_CAM_logit, local_heatmap, global_heatmap

    def discriminator_global(self, x_init, reuse=False, scope='discriminator_global'):
        #x_init=(32, 64, 64, 3)
        #3次卷积，维度放大，图片缩小
        #1次卷积，维度放大，图片不变
            #图片用两次不用的方法池化
            #池化结果进行1次全连接（两次全连接，所用的参数：权重矩阵，偏置矩阵 相同）
                #将两次全连接结果连接
            #将（权重+偏执）与x图片矩阵相乘
                #两次乘积相连接
        #对于两次图片乘积结果 进行两次卷积，不改变图片大小，维度缩小
        #输出 卷积图片（x=(32, 8, 8, 1)） ，池化连接结果（(32, 2)）
        with tf.variable_scope(scope, reuse=reuse):
            #经过6次卷积
            channel = self.ch#64
            x = conv(x_init, channel, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='conv_0')
            #卷积，x大小缩小一倍
            #x=(32, 32, 32, 64)
            x = lrelu(x, 0.2)
            for i in range(1, self.n_dis - 1):
                #x1=(32, 16, 16, 128)
                #x2=(32, 8, 8, 256)
                x = conv(x, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='conv_' + str(i))
                x = lrelu(x, 0.2)
                channel = channel * 2
            
            x = conv(x, channel * 2, kernel=4, stride=1, pad=1, pad_type='reflect', sn=self.sn, scope='conv_last')
            x = lrelu(x, 0.2)
            #此时步长变为了1，不缩小了
            #x=(32, 8, 8, 512)
            channel = channel * 2

            cam_x = global_avg_pooling(x)
            #对x进行全局平均池化
            #将原来的8*8*512的图片变为1*512的
            cam_gap_logit, cam_x_weight = fully_connected_with_w(cam_x, sn=self.sn, scope='CAM_logit')
            #全连接层，将1*521的图片归为1*1
            x_gap = tf.multiply(x, cam_x_weight)
            #原x与（全连接权重+偏执）相乘
            #cam_x_shape: (32, 512)
            #cam_gap_logit: (32, 1)
            #x_gap: (32, 8, 8, 512)
            
            cam_x = global_max_pooling(x)
            cam_gmp_logit, cam_x_weight = fully_connected_with_w(cam_x, sn=self.sn, reuse=True, scope='CAM_logit')
            x_gmp = tf.multiply(x, cam_x_weight)

            cam_logit = tf.concat([cam_gap_logit, cam_gmp_logit], axis=-1)
            #cam_logit：(32, 2)
            #将两次不同方法池化的1*1图片连接
            
            x = tf.concat([x_gap, x_gmp], axis=-1)
            #将两次全连接后的结果相连
            #x=(32, 8, 8, 1024)

            #经过两次大小不变的卷积，将图片的维度减小为1
            x = conv(x, channel, kernel=1, stride=1, scope='conv_1x1')
            #x=(32, 8, 8, 512)
            x = lrelu(x, 0.2)
            heatmap = tf.squeeze(tf.reduce_sum(x, axis=-1))

            x = conv(x, channels=1, kernel=4, stride=1, pad=1, pad_type='reflect', sn=self.sn, scope='D_logit')
            #x=(32, 8, 8, 1)
            return x, cam_logit, heatmap

    def discriminator_local(self, x_init, reuse=False, scope='discriminator_local'):
        #x_init=(32, 64, 64, 3)
        '''
        比discriminator_global 少了第一步的2次卷积
        '''
        #1次卷积，维度放大，图片缩小
        #1次卷积，维度放大，图片不变
            #图片用两次不用的方法池化
            #池化结果进行1次全连接（两次全连接，所用的参数：权重矩阵，偏置矩阵 相同）
                #将两次全连接结果连接
            #将（权重+偏执）与x图片矩阵相乘
                #两次乘积相连接
        #对于两次图片乘积结果 进行两次卷积，不改变图片大小，维度缩小
        #输出 卷积图片 ，池化连接结果    
        with tf.variable_scope(scope, reuse=reuse) :
            channel = self.ch
            x = conv(x_init, channel, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='conv_0')
            x = lrelu(x, 0.2)

            for i in range(1, self.n_dis - 2 - 1):
                x = conv(x, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='conv_' + str(i))
                print(str(i)+'times:',x.shape)
                x = lrelu(x, 0.2)

                channel = channel * 2

            print('x_conv:',x.shape)
            x = conv(x, channel * 2, kernel=4, stride=1, pad=1, pad_type='reflect', sn=self.sn, scope='conv_last')
            x = lrelu(x, 0.2)

            channel = channel * 2

            cam_x = global_avg_pooling(x)
            cam_gap_logit, cam_x_weight = fully_connected_with_w(cam_x, sn=self.sn, scope='CAM_logit')
            x_gap = tf.multiply(x, cam_x_weight)

            cam_x = global_max_pooling(x)
            cam_gmp_logit, cam_x_weight = fully_connected_with_w(cam_x, sn=self.sn, reuse=True, scope='CAM_logit')
            x_gmp = tf.multiply(x, cam_x_weight)

            cam_logit = tf.concat([cam_gap_logit, cam_gmp_logit], axis=-1)
            x = tf.concat([x_gap, x_gmp], axis=-1)

            x = conv(x, channel, kernel=1, stride=1, scope='conv_1x1')
            x = lrelu(x, 0.2)

            heatmap = tf.squeeze(tf.reduce_sum(x, axis=-1))

            x = conv(x, channels=1, kernel=4, stride=1, pad=1, pad_type='reflect', sn=self.sn, scope='D_logit')

            return x, cam_logit, heatmap

    ##################################################################################
    # Model
    ##################################################################################

    def generate_a2b(self, x_A, reuse=False):
        #翻译模型，将a翻译为b
        out, cam, _ ,img= self.generator(x_A, reuse=reuse, scope="generator_B")

        return out, cam,img

    def generate_b2a(self, x_B, reuse=False):
        out, cam, _ ,img= self.generator(x_B, reuse=reuse, scope="generator_A")

        return out, cam,img

    def discriminate_real(self, x_A, x_B):
        real_A_logit, real_A_cam_logit, _, _ = self.discriminator(x_A, scope="discriminator_A")
        real_B_logit, real_B_cam_logit, _, _ = self.discriminator(x_B, scope="discriminator_B")

        return real_A_logit, real_A_cam_logit, real_B_logit, real_B_cam_logit

    def discriminate_fake(self, x_ba, x_ab):
        fake_A_logit, fake_A_cam_logit, _, _ = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
        fake_B_logit, fake_B_cam_logit, _, _ = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

        return fake_A_logit, fake_A_cam_logit, fake_B_logit, fake_B_cam_logit

    def gradient_panalty(self, real, fake, scope="discriminator_A"):
        if self.gan_type.__contains__('dragan'):
            eps = tf.random_uniform(shape=tf.shape(real), minval=0., maxval=1.)
            _, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region

            fake = real + 0.5 * x_std * eps

        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolated = real + alpha * (fake - real)

        logit, cam_logit, _, _ = self.discriminator(interpolated, reuse=True, scope=scope)


        GP = []
        cam_GP = []

        for i in range(2) :
            grad = tf.gradients(logit[i], interpolated)[0] # gradient of D(interpolated)
            grad_norm = tf.norm(flatten(grad), axis=1) # l2 norm

            # WGAN - LP
            if self.gan_type == 'wgan-lp' :
                GP.append(self.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.))))

            elif self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
                GP.append(self.ld * tf.reduce_mean(tf.square(grad_norm - 1.)))

        for i in range(2) :
            grad = tf.gradients(cam_logit[i], interpolated)[0] # gradient of D(interpolated)
            grad_norm = tf.norm(flatten(grad), axis=1) # l2 norm

            # WGAN - LP
            if self.gan_type == 'wgan-lp' :
                cam_GP.append(self.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.))))

            elif self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
                cam_GP.append(self.ld * tf.reduce_mean(tf.square(grad_norm - 1.)))


        return sum(GP), sum(cam_GP)

    def build_model(self):
        if self.phase == 'train' :
            #初始化步长
            self.lr = tf.placeholder(tf.float32, name='learning_rate')


            """ Input Image"""
            Image_Data_Class = ImageData(self.img_size, self.img_ch, self.augment_flag)
            trainA = tf.data.Dataset.from_tensor_slices(self.trainA_dataset)
            trainB = tf.data.Dataset.from_tensor_slices(self.trainB_dataset)
            #将图片地址切片
            #print('ok trainA:',trainA)

            gpu_device = '/gpu:0'
            trainA = trainA.apply(shuffle_and_repeat(100)).apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)).apply(prefetch_to_device(gpu_device, None))
            trainB = trainB.apply(shuffle_and_repeat(100)).apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)).apply(prefetch_to_device(gpu_device, None))
            #打乱原图片排列
            #map_and_batch 1：将tensor的嵌套结构映射到另一个tensor嵌套结构的函数
            #              2：要在此数据集合并的单个batch中的连续元素数（一个batch 32 个元素，即输出是32维的元素）
            #              3：要并行创建的batch数。一方面，较高的值可以帮助减轻落后者的影响。另一方面，如果CPU空闲，较高的值可能会增加竞争。
            #              4：表示是否应丢弃最后一个batch，以防其大小小于所需值
            #返回 32*64*64*3 的随机增强的数组
            #应用gpu加速
            #print('ok trainA:',trainA.shape)
            trainA_iterator = trainA.make_one_shot_iterator()
            trainB_iterator = trainB.make_one_shot_iterator()

            self.domain_A = trainA_iterator.get_next()
            self.domain_B = trainB_iterator.get_next()

            """ Define Generator, Discriminator """
            x_ab, cam_ab = self.generate_a2b(self.domain_A) # real a
            #self.domain_A 是 卡通图片
            #x_ab是由self.domain_A经过下采样 再 上采样得到的图片
            x_ba, cam_ba = self.generate_b2a(self.domain_B) # real b
            #generate_a2b和generate_b2a是两套不同的参数
            
            
            x_aba, _ = self.generate_b2a(x_ab, reuse=True) # real b
            x_bab, _ = self.generate_a2b(x_ba, reuse=True) # real a
            #固定参数不变，再将generate_a2b生成的图片，用generate_b2a生成一遍
            #generate_b2a 尝试 将真人图生成卡通图
            #generate_a2b 尝试 将卡通图生成真人图
            #可以看做将generate_a2b与generate_b2a作为逆变换
            
            x_aa, cam_aa = self.generate_b2a(self.domain_A, reuse=True) # fake b
            x_bb, cam_bb = self.generate_a2b(self.domain_B, reuse=True) # fake a
            #固定参数不变
            #***将卡通图生成卡通图
                #确保在generate_b2a和generate_a2b过程中，颜色区域是不变的
            real_A_logit, real_A_cam_logit, real_B_logit, real_B_cam_logit = self.discriminate_real(self.domain_A, self.domain_B)
            #鉴别 真卡通图 与 真真人图
            fake_A_logit, fake_A_cam_logit, fake_B_logit, fake_B_cam_logit = self.discriminate_fake(x_ba, x_ab)
            #鉴别 假卡通图 与 假真人图
            #输入的是生成器生成的图片
            #输出的是图片经过卷积的32*8*8*1的张量和池化结果的连接(32, 2)
    

            """ Define Loss """
            if self.gan_type.__contains__('wgan') or self.gan_type == 'dragan' :
                GP_A, GP_CAM_A = self.gradient_panalty(real=self.domain_A, fake=x_ba, scope="discriminator_A")
                GP_B, GP_CAM_B = self.gradient_panalty(real=self.domain_B, fake=x_ab, scope="discriminator_B")
            else :
                GP_A, GP_CAM_A  = 0, 0
                GP_B, GP_CAM_B = 0, 0
            
            #接下来是对于假图片判别器discriminate_fake 真图片判别器discriminate_real的损失计算
            #对判别器和生成器的损失计算
            
            '''
            一、对抗损失（T）  
                用的最小二乘损失
            '''
            #fake_A_logit[0],fake_A_logit[1]与1的平方差的和+fake_A_cam_logit[0],fake_A_cam_logit[1]与1的平方差的和
            #从生成器的角度看，真图片会被discriminate鉴别器判为0，生成的图片经过判别器（1-fake_*），也要尽量被判别为0            
            G_ad_loss_A = (generator_loss(self.gan_type, fake_A_logit) + generator_loss(self.gan_type, fake_A_cam_logit))
            #生成假卡通图的损失
            G_ad_loss_B = (generator_loss(self.gan_type, fake_B_logit) + generator_loss(self.gan_type, fake_B_cam_logit))
            #生成假真人图的损失
            #生成器的损失

            D_ad_loss_A = (discriminator_loss(self.gan_type, real_A_logit, fake_A_logit) + discriminator_loss(self.gan_type, real_A_cam_logit, fake_A_cam_logit) + GP_A + GP_CAM_A)
            #对卡通图的鉴别损失
            #discriminator_loss力求将真图片判断为1，假图片判断为0
            D_ad_loss_B = (discriminator_loss(self.gan_type, real_B_logit, fake_B_logit) + discriminator_loss(self.gan_type, real_B_cam_logit, fake_B_cam_logit) + GP_B + GP_CAM_B)
            #鉴别器的损失
            
            '''
            二、Cycle 损失(T)
                 the image should be successfully translated back to the original domain
            '''
            reconstruction_A = L1_loss(x_aba, self.domain_A) # reconstruction
            #由卡通图生成真人图再生成卡通图的损失
            reconstruction_B = L1_loss(x_bab, self.domain_B) # reconstruction
            #由真人图生成卡通图再生成真人图的损失
            
            '''
            三、Identity 损失
                确保在A B相互变化时，身份信息不丢失
            '''
            identity_A = L1_loss(x_aa, self.domain_A)
            identity_B = L1_loss(x_bb, self.domain_B)
            #将卡通图生成卡通图的损失

            '''
            四、CAM 损失
                利用辅助分类器，使得G和D知道在哪里进行强化变换
                在A变B时，热力图应该有明显显示
                在A变A时，热力图应该没有显示
            '''
            cam_A = cam_loss(source=cam_ba, non_source=cam_aa)
            #cam_ba是从真人图到卡通图的全连接（两次，用不同方法池化）
            #cam_aa是从卡通图到卡通图的全连接
            cam_B = cam_loss(source=cam_ab, non_source=cam_bb)

            
            #开始时的比重是如何决定的？？？
            #
            Generator_A_gan = self.adv_weight * G_ad_loss_A
            #1
            #网络由真人图生成卡通图的损失*相对的比重
            Generator_A_cycle = self.cycle_weight * reconstruction_B
            #10
            #self.generate_a2b(self.generate_b2a(self.domain_B), reuse=True)
            #由真人图生成卡通图再生成真人图的损失*相对的比重
            Generator_A_identity = self.identity_weight * identity_A
            #10
            #由卡通图生成卡通图的损失*相对的比重
            Generator_A_cam = self.cam_weight * cam_A
            #1000
            #从真人图到卡通图的全连接*相对的比重


            Generator_B_gan = self.adv_weight * G_ad_loss_B
            Generator_B_cycle = self.cycle_weight * reconstruction_A
            Generator_B_identity = self.identity_weight * identity_B
            Generator_B_cam = self.cam_weight * cam_B
            print('ok 5')


            Generator_A_loss = Generator_A_gan + Generator_A_cycle + Generator_A_identity + Generator_A_cam
            #所有生成卡通图的损失
            Generator_B_loss = Generator_B_gan + Generator_B_cycle + Generator_B_identity + Generator_B_cam


            Discriminator_A_loss = self.adv_weight * D_ad_loss_A
            #对生成的卡通图 和 真的卡通图的鉴别损失+生成的卡通图的全连接 和 真的卡通图的全连接的鉴别损失*权重
            Discriminator_B_loss = self.adv_weight * D_ad_loss_B

            self.Generator_loss = Generator_A_loss + Generator_B_loss + regularization_loss('generator')
            #生成器的总损失
            self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss + regularization_loss('discriminator')
            #鉴别器的总损失
            print('55')


            """ Result Image """
            #生成的假图片（用于储存）
            self.fake_A = x_ba
            self.fake_B = x_ab

            #输入的真图片
            self.real_A = self.domain_A
            self.real_B = self.domain_B

            self.imgba=imgba
            self.imgab=imgab
            """ Training """
            t_vars = tf.trainable_variables()
            G_vars = [var for var in t_vars if 'generator' in var.name]
            D_vars = [var for var in t_vars if 'discriminator' in var.name]

            self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
            self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)
            #var_list：在优化时每次要迭代更新的参数集合

            """" Summary """
            # self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
            # self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)

            # self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
            # self.G_A_gan = tf.summary.scalar("G_A_gan", Generator_A_gan)
            # self.G_A_cycle = tf.summary.scalar("G_A_cycle", Generator_A_cycle)
            # self.G_A_identity = tf.summary.scalar("G_A_identity", Generator_A_identity)
            # self.G_A_cam = tf.summary.scalar("G_A_cam", Generator_A_cam)

            # self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
            # self.G_B_gan = tf.summary.scalar("G_B_gan", Generator_B_gan)
            # self.G_B_cycle = tf.summary.scalar("G_B_cycle", Generator_B_cycle)
            # self.G_B_identity = tf.summary.scalar("G_B_identity", Generator_B_identity)
            # self.G_B_cam = tf.summary.scalar("G_B_cam", Generator_B_cam)

            # self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
            # self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

            '''
            画图
            '''
            # self.rho_var = []
            # for var in tf.trainable_variables():
                # if 'rho' in var.name:
                    # self.rho_var.append(tf.summary.histogram(var.name, var))
                    # self.rho_var.append(tf.summary.scalar(var.name + "_min", tf.reduce_min(var)))
                    # self.rho_var.append(tf.summary.scalar(var.name + "_max", tf.reduce_max(var)))
                    # self.rho_var.append(tf.summary.scalar(var.name + "_mean", tf.reduce_mean(var)))
            # print('ok 7')

            # g_summary_list = [self.G_A_loss, self.G_A_gan, self.G_A_cycle, self.G_A_identity, self.G_A_cam,
                              # self.G_B_loss, self.G_B_gan, self.G_B_cycle, self.G_B_identity, self.G_B_cam,
                              # self.all_G_loss]

            # g_summary_list.extend(self.rho_var)
            # d_summary_list = [self.D_A_loss, self.D_B_loss, self.all_D_loss]

            # self.G_loss = tf.summary.merge(g_summary_list)
            # self.D_loss = tf.summary.merge(d_summary_list)

        else :
            """ Test """
            self.test_domain_A = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.img_ch], name='test_domain_A')
            self.test_domain_B = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.img_ch], name='test_domain_B')


            self.test_fake_B, _ = self.generate_a2b(self.test_domain_A)
            self.test_fake_A, _ = self.generate_b2a(self.test_domain_B)

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()
        print('train 1')
        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        #self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)
        print('train 2')

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")
        print('train 3')
        # loop for epoch
        start_time = time.time()
        past_g_loss = -1.
        lr = self.init_lr
        #训练步长
        for epoch in range(start_epoch, self.epoch):
            # lr = self.init_lr if epoch < self.decay_epoch else self.init_lr * (self.epoch - epoch) / (self.epoch - self.decay_epoch)
            if self.decay_flag :
                #lr = self.init_lr * pow(0.5, epoch // self.decay_epoch)
                lr = self.init_lr if epoch < self.decay_epoch else self.init_lr * (self.epoch - epoch) / (self.epoch - self.decay_epoch)
            print('start_batch_id',start_batch_id)
            print('self.iteration',self.iteration) 
            for idx in range(start_batch_id, self.iteration):
                train_feed_dict = {
                    self.lr : lr
                }

                # Update D
                _, d_loss = self.sess.run([self.D_optim,
                                                        self.Discriminator_loss], feed_dict = train_feed_dict)
                print('ok d_loss')
                #self.writer.add_summary(summary_str, counter)

                # Update G
                g_loss = None
                if (counter - 1) % self.n_critic == 0 :
                #self.n_critic=1,这里是每一次都计算
                    
                    batch_A_images= self.sess.run([self.real_A], feed_dict = train_feed_dict)
                    batch_B_images= self.sess.run([self.real_B], feed_dict = train_feed_dict)
                    fake_A= self.sess.run([self.fake_A], feed_dict = train_feed_dict)
                    fake_B= self.sess.run([self.fake_B], feed_dict = train_feed_dict)
                    _ = self.sess.run([self.G_optim], feed_dict = train_feed_dict)#非常费时
                    g_loss = self.sess.run([self.Generator_loss], feed_dict = train_feed_dict)
                    #self.writer.add_summary(summary_str, counter)
                    past_g_loss = g_loss

                # display training status
                counter += 1
                if g_loss == None :
                    g_loss = past_g_loss
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f" % (epoch, idx, self.iteration, time.time() - start_time))
                print('d_loss:',d_loss)
                print('g_loss:',g_loss)
                if np.mod(idx+1, self.print_freq) == 0 :
                    save_images(batch_A_images, [self.batch_size, 1],
                                './{}/real_A_{:03d}_{:05d}.png'.format(self.  sample_dir, epoch, idx+1))
                    save_images(batch_B_images, [self.batch_size, 1],
                                './{}/real_B_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))

                    save_images(fake_A, [self.batch_size, 1],
                                './{}/fake_A_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))
                    save_images(fake_B, [self.batch_size, 1],
                                './{}/fake_B_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))
                                
                    save_images(imgab, [self.batch_size, 1],
                                './{}/imgab_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))
                    save_images(imgba, [self.batch_size, 1],
                                './{}/imgba_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))                                
                if np.mod(idx + 1, self.save_freq) == 0:
                    self.save(self.checkpoint_dir, counter)



            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model for final step
            self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        n_res = str(self.n_res) + 'resblock'
        n_dis = str(self.n_dis) + 'dis'

        if self.smoothing :
            smoothing = '_smoothing'
        else :
            smoothing = ''

        if self.sn :
            sn = '_sn'
        else :
            sn = ''

        return "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}{}{}".format(self.model_name, self.dataset_name,
                                                         self.gan_type, n_res, n_dis,
                                                         self.n_critic,
                                                         self.adv_weight, self.cycle_weight, self.identity_weight, self.cam_weight, sn, smoothing)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()
        test_A_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testA'))
        test_B_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testB'))

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        for sample_file  in test_A_files : # A -> B
            print('Processing A image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file, size=self.img_size))
            image_path = os.path.join(self.result_dir,'{0}'.format(os.path.basename(sample_file)))

            fake_img = self.sess.run(self.test_fake_B, feed_dict = {self.test_domain_A : sample_image})
            save_images(fake_img, [1, 1], image_path)

            index.write("<td>%s</td>" % os.path.basename(image_path))

            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                '../..' + os.path.sep + sample_file), self.img_size, self.img_size))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                '../..' + os.path.sep + image_path), self.img_size, self.img_size))
            index.write("</tr>")

        for sample_file  in test_B_files : # B -> A
            print('Processing B image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file, size=self.img_size))
            image_path = os.path.join(self.result_dir,'{0}'.format(os.path.basename(sample_file)))

            fake_img = self.sess.run(self.test_fake_A, feed_dict = {self.test_domain_B : sample_image})

            save_images(fake_img, [1, 1], image_path)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                    '../..' + os.path.sep + sample_file), self.img_size, self.img_size))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                    '../..' + os.path.sep + image_path), self.img_size, self.img_size))
            index.write("</tr>")
        index.close()
