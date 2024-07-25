
#include "csr_offsets.h"


void vtg_enable_write(u32 data){	    //stop IP
	write_u32(data,IO_APB_SLAVE_3_INPUT+CSR_VTG_ENABLE_ADDR);
}

void vtg_hres_write(u32 data){	
	write_u32(data,IO_APB_SLAVE_3_INPUT+CSR_VTG_HRES_ADDR);
}

void vtg_hsync_start_write(u32 data){
	write_u32(data,IO_APB_SLAVE_3_INPUT+CSR_VTG_HSYNC_START_ADDR);
}

void vtg_hsync_end_write(u32 data){
	write_u32(data,IO_APB_SLAVE_3_INPUT+CSR_VTG_HSYNC_END_ADDR);
}

void vtg_hscan_write(u32 data){
	write_u32(data,IO_APB_SLAVE_3_INPUT+CSR_VTG_HSCAN_ADDR);
}

void vtg_vres_write(u32 data){
	write_u32(data,IO_APB_SLAVE_3_INPUT+CSR_VTG_VRES_ADDR);
}

void vtg_vsync_start_write(u32 data){
	write_u32(data,IO_APB_SLAVE_3_INPUT+CSR_VTG_VSYNC_START_ADDR);
}

void vtg_vsync_end_write(u32 data){
	write_u32(data,IO_APB_SLAVE_3_INPUT+CSR_VTG_VSYNC_END_ADDR);
}

void vtg_vscan_write(u32 data){
	write_u32(data,IO_APB_SLAVE_3_INPUT+CSR_VTG_VSCAN_ADDR);
}
