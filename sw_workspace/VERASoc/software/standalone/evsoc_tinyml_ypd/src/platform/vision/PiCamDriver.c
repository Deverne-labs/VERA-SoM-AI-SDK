
#include "bsp.h"
#include "i2c.h"
#include "i2cDemo.h" //From BSP
#include "riscv.h"
#include "PiCamDriver.h"
#include "common.h"


struct config_word_t const cfg_simple_awb_[] =
{
	// Disable Advanced AWB
	{0x518d ,0x00},
	{0x518f ,0x20},
	{0x518e ,0x00},
	{0x5190 ,0x20},
	{0x518b ,0x00},
	{0x518c ,0x00},
	{0x5187 ,0x10},
	{0x5188 ,0x10},
	{0x5189 ,0x40},
	{0x518a ,0x40},
	{0x5186 ,0x10},
	{0x5181 ,0x58},
	{0x5184 ,0x25},
	{0x5182 ,0x11},

	// Enable simple AWB
	{0x3406 ,0x00},
	{0x5183 ,0x80},
	{0x5191 ,0xff},
	{0x5192 ,0x00},
	{0x5001 ,0x03}
};

struct config_word_t const cfg_disable_awb_[] =
{
	{0x5001 ,0x02}
};

struct config_word_t cfg_advanced_awb_[] =
{
	// Enable Advanced AWB
		{0x3400 ,0x05},  //R msb
		{0x3401 ,0x00},  //R lsb
		{0x3402 ,0x03},  //G msb
		{0x3403 ,0x90},  //G lsb
		{0x3404 ,0x03},  //B msb
		{0x3405 ,0x00},  //B lsb

	{0x3406 ,0x01},

	{0x5192 ,0x04},
	{0x5191 ,0xf8},
	{0x518d ,0x26},
	{0x518f ,0x42},
	{0x518e ,0x2b},
	{0x5190 ,0x42},
	{0x518b ,0xd0},
	{0x518c ,0xbd},
	{0x5187 ,0x18},
	{0x5188 ,0x18},
	{0x5189 ,0x56},
	{0x518a ,0x5c},
	{0x5186 ,0x1c},
	{0x5181 ,0x50},
	{0x5184 ,0x20},
	{0x5182 ,0x11},
	{0x5183 ,0x00},
	{0x5001 ,0x03}
};



struct config_word_t const cfg_1080p_30fps_[] =
	{//1920 x 1080 @ 30fps, RAW10, MIPISCLK=420, SCLK=84MHz, PCLK=84M
		//PLL1 configuration
		//[7:4]=0010 System clock divider /2, [3:0]=0001 Scale divider for MIPI /1
		{0x3035, 0x21}, // 30fps setting
		//[7:0]=105 PLL multiplier
		{0x3036, 0x69},
		//[4]=0 PLL root divider /1, [3:0]=5 PLL pre-divider /1.5
		{0x3037, 0x05},
		//[5:4]=01 PCLK root divider /2, [3:2]=00 SCLK2x root divider /1, [1:0]=01 SCLK root divider /2
		{0x3108, 0x11},

		//[6:4]=001 PLL charge pump, [3:0]=1010 MIPI 10-bit mode
		{0x3034, 0x1A},

		//[3:0]=0 X address start high byte
		{0x3800, (336 >> 8) & 0x0F},
		//[7:0]=0 X address start low byte
		{0x3801, 336 & 0xFF},
		//[2:0]=0 Y address start high byte
		{0x3802, (427 >> 8) & 0x07},
		//[7:0]=0 Y address start low byte
		{0x3803, 427 & 0xFF},

		//[3:0] X address end high byte
		{0x3804, (2287 >> 8) & 0x0F},
		//[7:0] X address end low byte
		{0x3805, 2287 & 0xFF},
		//[2:0] Y address end high byte
		{0x3806, (1530 >> 8) & 0x07},
		//[7:0] Y address end low byte
		{0x3807, 1530 & 0xFF},

		//[3:0]=0 timing hoffset high byte
		{0x3810, (12 >> 8) & 0x0F},
		//[7:0]=0 timing hoffset low byte
		{0x3811, 16 & 0xFF},
		//[2:0]=0 timing voffset high byte
		{0x3812, (12 >> 8) & 0x07},
		//[7:0]=0 timing voffset low byte
		{0x3813, 12 & 0xFF},

		//[3:0] Output horizontal width high byte
		{0x3808, (1920 >> 8) & 0x0F},
		//[7:0] Output horizontal width low byte
		{0x3809, 1920 & 0xFF},
		//[2:0] Output vertical height high byte
		{0x380a, (1080 >> 8) & 0x7F},
		//[7:0] Output vertical height low byte
		{0x380b, 1080 & 0xFF},

		//HTS line exposure time in # of pixels Tline=HTS/sclk
		{0x380c, (2500 >> 8) & 0x1F},
		{0x380d, 2500 & 0xFF},
		//VTS frame exposure time in # lines
		{0x380e, (1120 >> 8) & 0xFF},
		{0x380f, 1120 & 0xFF},

		//[7:4]=0x1 horizontal odd subsample increment, [3:0]=0x1 horizontal even subsample increment
		{0x3814, 0x11},
		//[7:4]=0x1 vertical odd subsample increment, [3:0]=0x1 vertical even subsample increment
		{0x3815, 0x11},

		//[2]=0 ISP mirror, [1]=0 sensor mirror, [0]=0 no horizontal binning
		{0x3821, 0x00},

		//little MIPI shit: global timing unit, period of PCLK in ns * 2(depends on # of lanes)
		{0x4837, 24}, // 1/84M*2

		//Undocumented anti-green settings
		{0x3618, 0x00}, // Removes vertical lines appearing under bright light
		{0x3612, 0x59},
		{0x3708, 0x64},
		{0x3709, 0x52},
		{0x370c, 0x03},

		//[7:4]=0x0 Formatter RAW, [3:0]=0x0 BGBG/GRGR
		{0x4300, 0x00},
		//[2:0]=0x3 Format select ISP RAW (DPC)
		{0x501f, 0x03}
	};

struct config_word_t cfg_720p_60fps_[] =
{//1280 x 720 binned, RAW10, MIPISCLK=280M, SCLK=56Mz, PCLK=56M
	//PLL1 configuration
	/*//[7:4]=0010 System clock divider /2, [3:0]=0001 Scale divider for MIPI /1
	{0x3035, 0x21},
	//[7:0]=70 PLL multiplier
	{0x3036, 0x46},
	//[4]=0 PLL root divider /1, [3:0]=5 PLL pre-divider /1.5
	{0x3037, 0x05},
	//[5:4]=01 PCLK root divider /2, [3:2]=00 SCLK2x root divider /1, [1:0]=01 SCLK root divider /2
	{0x3108, 0x11},*/

	// PLL1 configuration
	// [7:4]=0100 System clock divider /4, [3:0]=0001 Scale divider for MIPI /1
	{0x3035, 0x41},
	// [7:0]=105 PLL multiplier
	{0x3036, 0x69},
	// [4]=0 PLL root divider /1, [3:0]=5 PLL pre-divider /1.5
	{0x3037, 0x05},
	// [5:4]=01 PCLK root divider /2, [3:2]=00 SCLK2x root divider /1, [1:0]=01 SCLK root divider /2
	{0x3108, 0x11},

	//[6:4]=001 PLL charge pump, [3:0]=1010 MIPI 10-bit mode
	{0x3034, 0x1A},

	//[3:0]=0 X address start high byte
	{0x3800, (0 >> 8) & 0x0F},
	//[7:0]=0 X address start low byte
	{0x3801, 0 & 0xFF},
	//[2:0]=0 Y address start high byte
	{0x3802, (9 >> 8) & 0x07},
	//[7:0]=0 Y address start low byte
	{0x3803, 9 & 0xFF},

	//[3:0] X address end high byte
	{0x3804, (2619 >> 8) & 0x0F},
	//[7:0] X address end low byte
	{0x3805, 2619 & 0xFF},
	//[2:0] Y address end high byte
	{0x3806, (1948 >> 8) & 0x07},
	//[7:0] Y address end low byte
	{0x3807, 1948 & 0xFF},

	//[3:0]=0 timing hoffset high byte
	{0x3810, (0 >> 8) & 0x0F},
	//[7:0]=0 timing hoffset low byte
	{0x3811, 0 & 0xFF},
	//[2:0]=0 timing voffset high byte
	{0x3812, (0 >> 8) & 0x07},
	//[7:0]=0 timing voffset low byte
	{0x3813, 0 & 0xFF},

	//[3:0] Output horizontal width high byte
	{0x3808, (1280 >> 8) & 0x0F},
	//{0x3808, (1920 >> 8) & 0x0F},
	//[7:0] Output horizontal width low byte
	{0x3809, 1280 & 0xFF},
	//{0x3809, 1920 & 0xFF},
	//[2:0] Output vertical height high byte
	{0x380a, (720 >> 8) & 0x7F},
	//{0x380a, (1080 >> 8) & 0x7F},
	//[7:0] Output vertical height low byte
	{0x380b, 720 & 0xFF},
	//{0x380b, 1080 & 0xFF},

	//HTS line exposure time in # of pixels
	{0x380c, (1896 >> 8) & 0x1F},
	{0x380d, 1896 & 0xFF},
	//VTS frame exposure time in # lines
	{0x380e, (984 >> 8) & 0xFF},
	{0x380f, 984 & 0xFF},

	//[7:4]=0x3 horizontal odd subsample increment, [3:0]=0x1 horizontal even subsample increment
	{0x3814, 0x31},
	//[7:4]=0x3 vertical odd subsample increment, [3:0]=0x1 vertical even subsample increment
	{0x3815, 0x31},

	//[2]=0 ISP mirror, [1]=0 sensor mirror, [0]=1 horizontal binning
	{0x3821, 0x01},

	//little MIPI shit: global timing unit, period of PCLK in ns * 2(depends on # of lanes)
	{0x4837, 36}, // 1/56M*2

	//black level calibration
	{0x4000, 0x1},
	{0x4002, 0x0},
	{0x4003, 0xBF},
	{0x4005, 0x2},
	{0x4009, 0x5},

	//Undocumented anti-green settings
	{0x3618, 0x00}, // Removes vertical lines appearing under bright light
	{0x3612, 0x59},
	{0x3708, 0x64},
	{0x3709, 0x52},
	{0x370c, 0x03},

	//[7:4]=0x0 Formatter RAW, [3:0]=0x0 BGBG/GRGR
	{0x4300, 0x00},
	//[2:0]=0x3 Format select ISP RAW (DPC)
	{0x501f, 0x03}
};



struct config_word_t cfg_360p_60fps_[] =
{//1280 x 720 binned, RAW10, MIPISCLK=280M, SCLK=56Mz, PCLK=56M
	//PLL1 configuration
	//[7:4]=0010 System clock divider /2, [3:0]=0001 Scale divider for MIPI /1
	{0x3035, 0x21},
	//[7:0]=70 PLL multiplier
	{0x3036, 0x46},
	//[4]=0 PLL root divider /1, [3:0]=5 PLL pre-divider /1.5
	{0x3037, 0x05},
	//[5:4]=01 PCLK root divider /2, [3:2]=00 SCLK2x root divider /1, [1:0]=01 SCLK root divider /2
	{0x3108, 0x11},

	//[6:4]=001 PLL charge pump, [3:0]=1010 MIPI 10-bit mode
	{0x3034, 0x1A},

	//[3:0]=0 X address start high byte
	{0x3800, (0 >> 8) & 0x0F},
	//[7:0]=0 X address start low byte
	{0x3801, 0 & 0xFF},
	//[2:0]=0 Y address start high byte
	{0x3802, (9 >> 8) & 0x07},
	//[7:0]=0 Y address start low byte
	{0x3803, 9 & 0xFF},

	//[3:0] X address end high byte
	{0x3804, (2619 >> 8) & 0x0F},
	//[7:0] X address end low byte
	{0x3805, 2619 & 0xFF},
	//[2:0] Y address end high byte
	{0x3806, (1948 >> 8) & 0x07},
	//[7:0] Y address end low byte
	{0x3807, 1948 & 0xFF},

	//[3:0]=0 timing hoffset high byte
	{0x3810, (0 >> 8) & 0x0F},
	//[7:0]=0 timing hoffset low byte
	{0x3811, 0 & 0xFF},
	//[2:0]=0 timing voffset high byte
	{0x3812, (0 >> 8) & 0x07},
	//[7:0]=0 timing voffset low byte
	{0x3813, 0 & 0xFF},

	//[3:0] Output horizontal width high byte
	{0x3808, (640 >> 8) & 0x0F},
	//{0x3808, (1920 >> 8) & 0x0F},
	//[7:0] Output horizontal width low byte
	{0x3809, 640 & 0xFF},
	//{0x3809, 1920 & 0xFF},
	//[2:0] Output vertical height high byte
	{0x380a, (360 >> 8) & 0x7F},
	//{0x380a, (1080 >> 8) & 0x7F},
	//[7:0] Output vertical height low byte
	{0x380b, 360 & 0xFF},
	//{0x380b, 1080 & 0xFF},

	//HTS line exposure time in # of pixels
	{0x380c, (1896 >> 8) & 0x1F},
	{0x380d, 1896 & 0xFF},
	//VTS frame exposure time in # lines
	{0x380e, (984 >> 8) & 0xFF},
	{0x380f, 984 & 0xFF},

	//[7:4]=0x3 horizontal odd subsample increment, [3:0]=0x1 horizontal even subsample increment
	{0x3814, 0x31},
	//[7:4]=0x3 vertical odd subsample increment, [3:0]=0x1 vertical even subsample increment
	{0x3815, 0x31},

	//[2]=0 ISP mirror, [1]=0 sensor mirror, [0]=1 horizontal binning
	{0x3821, 0x01},

	//little MIPI shit: global timing unit, period of PCLK in ns * 2(depends on # of lanes)
	{0x4837, 36}, // 1/56M*2

	//Undocumented anti-green settings
	{0x3618, 0x00}, // Removes vertical lines appearing under bright light
	{0x3612, 0x59},
	{0x3708, 0x64},
	{0x3709, 0x52},
	{0x370c, 0x03},

	//[7:4]=0x0 Formatter RAW, [3:0]=0x0 BGBG/GRGR
	{0x4300, 0x00},
	//[2:0]=0x3 Format select ISP RAW (DPC)
	{0x501f, 0x03}
};


struct config_word_t cfg_init_[] =
{
	//[7]=0 Software reset; [6]=1 Software power down; Default=0x02
	{0x3008, 0x42},
	//[1]=1 System input clock from PLL; Default read = 0x11
	{0x3103, 0x03},
	//[3:0]=0000 MD2P,MD2N,MCP,MCN input; Default=0x00
	{0x3017, 0x00},
	//[7:2]=000000 MD1P,MD1N, D3:0 input; Default=0x00
	{0x3018, 0x00},
	//[6:4]=001 PLL charge pump, [3:0]=1000 MIPI 8-bit mode
	{0x3034, 0x18},


	//PLL1 configuration
	//[7:4]=0001 System clock divider /1, [3:0]=0001 Scale divider for MIPI /1
	{0x3035, 0x11},
	//[7:0]=56 PLL multiplier
	{0x3036, 0x38},
	//[4]=1 PLL root divider /2, [3:0]=1 PLL pre-divider /1
	{0x3037, 0x11},
	//[5:4]=00 PCLK root divider /1, [3:2]=00 SCLK2x root divider /1, [1:0]=01 SCLK root divider /2
	{0x3108, 0x01},
	//PLL2 configuration
	//[5:4]=01 PRE_DIV_SP /1.5, [2]=1 R_DIV_SP /1, [1:0]=00 DIV12_SP /1
	{0x303D, 0x10},
	//[4:0]=11001 PLL2 multiplier DIV_CNT5B = 25
	{0x303B, 0x19},

	{0x3630, 0x2e},
	{0x3631, 0x0e},
	{0x3632, 0xe2},
	{0x3633, 0x23},
	{0x3621, 0xe0},
	{0x3704, 0xa0},
	{0x3703, 0x5a},
	{0x3715, 0x78},
	{0x3717, 0x01},
	{0x370b, 0x60},
	{0x3705, 0x1a},
	{0x3905, 0x02},
	{0x3906, 0x10},
	{0x3901, 0x0a},
	{0x3731, 0x02},
	//VCM debug mode
	{0x3600, 0x37},
	{0x3601, 0x33},
	//System control register changing not recommended
	{0x302d, 0x60},
	//??
	{0x3620, 0x52},
	{0x371b, 0x20},
	//?? DVP
	{0x471c, 0x50},

	{0x3a13, 0x43},
	{0x3a18, 0x00},
	{0x3a19, 0xf8},
	{0x3635, 0x13},
	{0x3636, 0x06},
	{0x3634, 0x44},
	{0x3622, 0x01},
	{0x3c01, 0x34},
	{0x3c04, 0x28},
	{0x3c05, 0x98},
	{0x3c06, 0x00},
	{0x3c07, 0x08},
	{0x3c08, 0x00},
	{0x3c09, 0x1c},
	{0x3c0a, 0x9c},
	{0x3c0b, 0x40},

	//[7]=1 color bar enable, [3:2]=00 eight color bar
	{0x503d, 0x00},
	//[2]=1 ISP vflip, [1]=1 sensor vflip
	{0x3820, 0x46},

	//[7:5]=010 Two lane mode, [4]=0 MIPI HS TX no power down, [3]=0 MIPI LP RX no power down, [2]=1 MIPI enable, [1:0]=10 Debug mode; Default=0x58
	{0x300e, 0x45},
	//[5]=0 Clock free running, [4]=1 Send line short packet, [3]=0 Use lane1 as default, [2]=1 MIPI bus LP11 when no packet; Default=0x04
	{0x4800, 0x14},
	{0x302e, 0x08},
	//[7:4]=0x3 YUV422, [3:0]=0x0 YUYV
	//{0x4300, 0x30},
	//[7:4]=0x6 RGB565, [3:0]=0x0 {b[4:0],g[5:3],g[2:0],r[4:0]}
	{0x4300, 0x6f},
	{0x501f, 0x01},

	{0x4713, 0x03},
	{0x4407, 0x04},
	{0x440e, 0x00},
	{0x460b, 0x35},
	//[1]=0 DVP PCLK divider manual control by 0x3824[4:0]
	{0x460c, 0x20},
	//[4:0]=1 SCALE_DIV=INT(3824[4:0]/2)
	{0x3824, 0x01},

	//[7]=1 LENC correction enabled, [5]=1 RAW gamma enabled, [2]=1 Black pixel cancellation enabled, [1]=1 White pixel cancellation enabled, [0]=1 Color interpolation enabled
	{0x5000, 0x07},
	//[7]=0 Special digital effects, [5]=0 scaling, [2]=0 UV average disabled, [1]=1 Color matrix enabled, [0]=1 Auto white balance enabled
	{0x5001, 0x03}
};




int PiCam_WriteRegData(u16 reg,u8 data)
{
	u8 outdata;

    i2c_masterStartBlocking(I2C_CTRL_MIPI);

    i2c_txByte(I2C_CTRL_MIPI, i2c_cam_addr);
	i2c_txNackBlocking(I2C_CTRL_MIPI);
	if (assert_pcam(i2c_rxAck(I2C_CTRL_MIPI)) ) // Optional check
		return 1;
	i2c_txByte(I2C_CTRL_MIPI, (reg>>8) & 0xFF);
	i2c_txNackBlocking(I2C_CTRL_MIPI);
	if (assert_pcam(i2c_rxAck(I2C_CTRL_MIPI)) ) // Optional check
		return 1;
	i2c_txByte(I2C_CTRL_MIPI, (reg) & 0xFF);
	i2c_txNackBlocking(I2C_CTRL_MIPI);
	if (assert_pcam(i2c_rxAck(I2C_CTRL_MIPI)) ) // Optional check
		return 1;
	i2c_txByte(I2C_CTRL_MIPI, data & 0xFF);
	i2c_txNackBlocking(I2C_CTRL_MIPI);
	if (assert_pcam(i2c_rxAck(I2C_CTRL_MIPI)) ) // Optional check
		return 1;
	i2c_masterStopBlocking(I2C_CTRL_MIPI);

	return 0;
}

u8 PiCam_ReadRegData(u16 reg)
{
	u8 outdata;

    i2c_masterStartBlocking(I2C_CTRL_MIPI);

    i2c_txByte(I2C_CTRL_MIPI, i2c_cam_addr);
	i2c_txNackBlocking(I2C_CTRL_MIPI);
	assert_pcam(i2c_rxAck(I2C_CTRL_MIPI)); // Optional check

	i2c_txByte(I2C_CTRL_MIPI, (reg>>8) & 0xFF);
	i2c_txNackBlocking(I2C_CTRL_MIPI);
	assert_pcam(i2c_rxAck(I2C_CTRL_MIPI)); // Optional check

	i2c_txByte(I2C_CTRL_MIPI, (reg) & 0xFF);
	i2c_txNackBlocking(I2C_CTRL_MIPI);
	assert_pcam(i2c_rxAck(I2C_CTRL_MIPI)); // Optional check

	i2c_masterStopBlocking(I2C_CTRL_MIPI);
	i2c_masterStartBlocking(I2C_CTRL_MIPI);

	i2c_txByte(I2C_CTRL_MIPI, (i2c_cam_addr) | 0x01);
	i2c_txNackBlocking(I2C_CTRL_MIPI);
	assert_pcam(i2c_rxAck(I2C_CTRL_MIPI)); // Optional check

	i2c_txByte(I2C_CTRL_MIPI, 0xFF);
	i2c_txNackBlocking(I2C_CTRL_MIPI);
	assert_pcam(i2c_rxNack(I2C_CTRL_MIPI)); // Optional check
	outdata = i2c_rxData(I2C_CTRL_MIPI);

	i2c_masterStopBlocking(I2C_CTRL_MIPI);

	return outdata;
}


void AccessCommSeq(void)
{
	PiCam_WriteRegData(0x30EB, 0x05);
	PiCam_WriteRegData(0x30EB, 0x0C);
	PiCam_WriteRegData(0x300A, 0xFF);
	PiCam_WriteRegData(0x300B, 0xFF);
	PiCam_WriteRegData(0x30EB, 0x05);
	PiCam_WriteRegData(0x30EB, 0x09);
}

void PiCam_Output_Size(u16 X,u16 Y)
{
	PiCam_WriteRegData(x_output_size_A_1	, X>>8);
	PiCam_WriteRegData(x_output_size_A_0	, X & 0xFF);
	PiCam_WriteRegData(y_output_size_A_1	, Y>>8);
	PiCam_WriteRegData(y_output_size_A_0	, Y & 0xFF);
}

void PiCam_Output_activePixel(u16 XStart,u16 XEnd, u16 YStart, u16 YEnd)
{

	//Max Active pixel 3280* 2464--imx219

	PiCam_WriteRegData(X_ADD_STA_A_1	, XStart>>8);
	PiCam_WriteRegData(X_ADD_STA_A_0	, XStart&0xFF);
	PiCam_WriteRegData(X_ADD_END_A_1	, XEnd>>8);
	PiCam_WriteRegData(X_ADD_END_A_0	, XEnd&0xFF);

	PiCam_WriteRegData(Y_ADD_STA_A_1	, YStart>>8);
	PiCam_WriteRegData(Y_ADD_STA_A_0	, YStart&0xFF);
	PiCam_WriteRegData(Y_ADD_END_A_1	, YEnd>>8);
	PiCam_WriteRegData(Y_ADD_END_A_0	, YEnd&0xFF);
}

void PiCam_Output_activePixelX(u16 XStart,u16 XEnd)
{
	//Max Active pixel 3280* 2464--imx219

	PiCam_WriteRegData(X_ADD_STA_A_1	, XStart>>8);
	PiCam_WriteRegData(X_ADD_STA_A_0	, XStart&0xFF);
	PiCam_WriteRegData(X_ADD_END_A_1	, XEnd>>8);
	PiCam_WriteRegData(X_ADD_END_A_0	, XEnd&0xFF);
}

void PiCam_Output_activePixelY(u16 YStart,u16 YEnd)
{
	//Max Active pixel 3280* 2464--imx219

	PiCam_WriteRegData(Y_ADD_STA_A_1	, YStart>>8);
	PiCam_WriteRegData(Y_ADD_STA_A_0	, YStart&0xFF);
	PiCam_WriteRegData(Y_ADD_END_A_1	, YEnd>>8);
	PiCam_WriteRegData(Y_ADD_END_A_0	, YEnd&0xFF);
}

void PiCam_SetBinningMode(u8 Xmode, u8 Ymode)
{
	//0:no-binning
	//1:x2-binning
	//2:x4-binning
	//3:x2 analog (special)

	if(Xmode>=3)	Xmode=3;
	if(Ymode>=3)	Ymode=3;

	PiCam_WriteRegData(BINNING_MODE_H_A, Xmode);
	PiCam_WriteRegData(BINNING_MODE_V_A, Ymode);
}

void PiCam_Output_ColorBarSize(u16 X,u16 Y)
{
	PiCam_WriteRegData(TP_WINDOW_WIDTH_1	, X>>8);
	PiCam_WriteRegData(TP_WINDOW_WIDTH_0	, X & 0xFF);
	PiCam_WriteRegData(TP_WINDOW_HEIGHT_1	, Y>>8);
	PiCam_WriteRegData(TP_WINDOW_HEIGHT_0	, Y & 0xFF);
}

void PiCam_TestPattern(u8 Enable,u8 mode,u16 X,u16 Y)
{
	//0000h - no pattern (default)
	//0001h - solid color
	//0002h - 100 % color bars
	//0003h - fade to grey color bar
	//0004h - PN9
	//0005h - 16 split color bar
	//0006h - 16 split inverted color bar
	//0007h - column counter
	//0008h - inverted column counter
	//0009h - PN31

	PiCam_WriteRegData(test_pattern_Ena, 0x00);

	if(Enable==0)	mode=0;

	PiCam_WriteRegData(test_pattern_mode, mode);

	PiCam_Output_ColorBarSize(X,Y);
}

void PiCam_Gainfilter(u8 AGain, u16 DGain)
{
	PiCam_WriteRegData(ANA_GAIN_GLOBAL_A, AGain&0xFF);
	PiCam_WriteRegData(DIG_GAIN_GLOBAL_A_1, (DGain>>8)&0x0F);
	PiCam_WriteRegData(DIG_GAIN_GLOBAL_A_0, DGain&0xFF);
}






void PCam5C_init(){
	u8 id_h, id_l;
	id_h = PiCam_ReadRegData(reg_ID_h);
	id_l = PiCam_ReadRegData(reg_ID_l);

	//MicroPrintf("\n\r");
	//MicroPrintf("PCam5C ID : %02x %02x\n\r",id_h, id_l);

	if (id_h != dev_ID_h_ || id_l != dev_ID_l_)
	{
		//MicroPrintf("PCam5C Initial Error !\n\r");
		//MicroPrintf("Got %02x %02x. Expected %02x %02x\r\n", id_h, id_l, dev_ID_h_, dev_ID_l_);
	}


	//[1]=0 System input clock from pad; Default read = 0x11
	PiCam_WriteRegData(0x3103, 0x11);
	//[7]=1 Software reset; [6]=0 Software power down; Default=0x02
	PiCam_WriteRegData(0x3008, 0x82);

	bsp_uDelay(1000000);

	for (size_t i=0; i<sizeof(cfg_init_)/sizeof(cfg_init_[0]); ++i)
	{
		PiCam_WriteRegData(cfg_init_[i].addr, cfg_init_[i].data);
	}


}


void PCam5C_config(){
	//[7]=0 Software reset; [6]=1 Software power down; Default=0x02
	PiCam_WriteRegData(0x3008, 0x42);

	for (size_t i=0; i<sizeof(cfg_720p_60fps_)/sizeof(cfg_720p_60fps_[0]); ++i)
	{
		PiCam_WriteRegData(cfg_720p_60fps_[i].addr, cfg_720p_60fps_[i].data);
	}
/*
	for (size_t i=0; i<sizeof(cfg_1080p_30fps_)/sizeof(cfg_1080p_30fps_[0]); ++i)
	{
		PiCam_WriteRegData(cfg_1080p_30fps_[i].addr, cfg_1080p_30fps_[i].data);
	}*/


	//[7]=0 Software reset; [6]=0 Software power down; Default=0x02
	PiCam_WriteRegData(0x3008, 0x02);

}

void PCam5C_set_awb(){
	//[7]=0 Software reset; [6]=1 Software power down; Default=0x02
	PiCam_WriteRegData(0x3008, 0x42);

	for (size_t i=0; i<sizeof(cfg_disable_awb_)/sizeof(cfg_disable_awb_[0]); ++i)
	{
		PiCam_WriteRegData(cfg_disable_awb_[i].addr, cfg_disable_awb_[i].data);
	}

	//[7]=0 Software reset; [6]=0 Software power down; Default=0x02
	PiCam_WriteRegData(0x3008, 0x02);
}






/*
void PiCam_init(void)
{

	   PiCam_WriteRegData(mode_select, 0x00);
	   AccessCommSeq();
	   PiCam_WriteRegData(CSI_LANE_MODE, 0x01);
	   PiCam_WriteRegData(DPHY_CTRL, 0x00);
	   PiCam_WriteRegData(EXCK_FREQ_1, 0x18);
	   PiCam_WriteRegData(EXCK_FREQ_0, 0x00);
	   PiCam_WriteRegData(FRM_LENGTH_A_1, 0x04);
	   PiCam_WriteRegData(FRM_LENGTH_A_0, 0x59);

	   PiCam_WriteRegData(LINE_LENGTH_A_1, 0x0D);
	   PiCam_WriteRegData(LINE_LENGTH_A_0, 0x78);

	   //PiCam_Output_activePixel(0, 3279, 0, 2463);
	   //PiCam_Output_activePixel(1020, 2300, 640, 1920); //Use offset to have central view for 1920 frame width
	   //PiCam_Output_activePixel(0, 2560, 0, 2560); //Use offset to have central view for 1920 frame width
	   PiCam_Output_activePixel(1020, 2900, 340, 2560); //Use offset to have central view for 1920 frame width

	   PiCam_Output_Size(640, 640);
	   //PiCam_Output_Size(1280, 720);
	   //PiCam_Output_Size(640, 480);

	   PiCam_WriteRegData(X_ODD_INC_A, 0x01);
	   PiCam_WriteRegData(Y_ODD_INC_A, 0x01);

	   //0: No binning; 1: x2 binning; 2: x4 binning; 3: x2 binning (analog special)
	   PiCam_SetBinningMode(1, 1);

	   PiCam_WriteRegData(CSI_DATA_FORMAT_A_1, 0x0A);
	   PiCam_WriteRegData(CSI_DATA_FORMAT_A_0, 0x0A);

	   PiCam_WriteRegData(VTPXCK_DIV, 0x05);
	   PiCam_WriteRegData(VTSYCK_DIV, 0x01);
	   PiCam_WriteRegData(PREPLLCK_VT_DIV, 0x03);
	   PiCam_WriteRegData(PREPLLCK_OP_DIV, 0x03);
	   PiCam_WriteRegData(PLL_VT_MPY_1, 0x00);
	   PiCam_WriteRegData(PLL_VT_MPY_0, 0x39);
	   PiCam_WriteRegData(OPPXCK_DIV, 0x0A);
	   PiCam_WriteRegData(OPSYCK_DIV, 0x01);
	   PiCam_WriteRegData(PLL_OP_MPY_1, 0x00);
	   PiCam_WriteRegData(PLL_OP_MPY_0, 0x72);

	   PiCam_WriteRegData(OPPXCK_DIV, 0x0A);
	   PiCam_WriteRegData(OPSYCK_DIV, 0x01);
	   PiCam_WriteRegData(PLL_OP_MPY_1, 0x00);
	   PiCam_WriteRegData(PLL_OP_MPY_0, 0x72);

	   PiCam_WriteRegData(mode_select, 0x01);

	   PiCam_Gainfilter(0xB9, 0x200);

	   PiCam_WriteRegData(LINE_LENGTH_A_1, 0x0D);
	   PiCam_WriteRegData(LINE_LENGTH_A_0, 0x78);


	   //Longer camera exposure time. Trade-off with lower frame rate. 30fps
	   PiCam_WriteRegData(FRM_LENGTH_A_1, 0x06);
	   PiCam_WriteRegData(FRM_LENGTH_A_0, 0xE3);
	   PiCam_WriteRegData(COARSE_INTEGRATION_TIME_A_1, 0x04);
	   PiCam_WriteRegData(COARSE_INTEGRATION_TIME_A_0, 0x14);


	   PiCam_WriteRegData(IMG_ORIENTATION_A, 0x00);


}
*/


