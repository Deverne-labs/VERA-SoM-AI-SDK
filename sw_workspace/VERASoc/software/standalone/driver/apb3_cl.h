////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2013-2023 Efinix Inc. All rights reserved.              
//
// This   document  contains  proprietary information  which   is        
// protected by  copyright. All rights  are reserved.  This notice       
// refers to original work by Efinix, Inc. which may be derivitive       
// of other work distributed under license of the authors.  In the       
// case of derivative work, nothing in this notice overrides the         
// original author's license agreement.  Where applicable, the           
// original license agreement is included in it's original               
// unmodified form immediately below this header.                        
//                                                                       
// WARRANTY DISCLAIMER.                                                  
//     THE  DESIGN, CODE, OR INFORMATION ARE PROVIDED “AS IS” AND        
//     EFINIX MAKES NO WARRANTIES, EXPRESS OR IMPLIED WITH               
//     RESPECT THERETO, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES,  
//     INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF          
//     MERCHANTABILITY, NON-INFRINGEMENT AND FITNESS FOR A PARTICULAR    
//     PURPOSE.  SOME STATES DO NOT ALLOW EXCLUSIONS OF AN IMPLIED       
//     WARRANTY, SO THIS DISCLAIMER MAY NOT APPLY TO LICENSEE.           
//                                                                       
// LIMITATION OF LIABILITY.                                              
//     NOTWITHSTANDING ANYTHING TO THE CONTRARY, EXCEPT FOR BODILY       
//     INJURY, EFINIX SHALL NOT BE LIABLE WITH RESPECT TO ANY SUBJECT    
//     MATTER OF THIS AGREEMENT UNDER TORT, CONTRACT, STRICT LIABILITY   
//     OR ANY OTHER LEGAL OR EQUITABLE THEORY (I) FOR ANY INDIRECT,      
//     SPECIAL, INCIDENTAL, EXEMPLARY OR CONSEQUENTIAL DAMAGES OF ANY    
//     CHARACTER INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF      
//     GOODWILL, DATA OR PROFIT, WORK STOPPAGE, OR COMPUTER FAILURE OR   
//     MALFUNCTION, OR IN ANY EVENT (II) FOR ANY AMOUNT IN EXCESS, IN    
//     THE AGGREGATE, OF THE FEE PAID BY LICENSEE TO EFINIX HEREUNDER    
//     (OR, IF THE FEE HAS BEEN WAIVED, $100), EVEN IF EFINIX SHALL HAVE 
//     BEEN INFORMED OF THE POSSIBILITY OF SUCH DAMAGES.  SOME STATES DO 
//     NOT ALLOW THE EXCLUSION OR LIMITATION OF INCIDENTAL OR            
//     CONSEQUENTIAL DAMAGES, SO THIS LIMITATION AND EXCLUSION MAY NOT   
//     APPLY TO LICENSEE.                                                
//
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "bsp.h"
#include "io.h"

#define EXAMPLE_APB3_SLV_REG0_OFFSET 	0
#define EXAMPLE_APB3_SLV_REG1_OFFSET 	4
#define EXAMPLE_APB3_SLV_REG2_OFFSET 	8
#define EXAMPLE_APB3_SLV_REG3_OFFSET 	12
#define EXAMPLE_APB3_SLV_REG4_OFFSET 	16
#define EXAMPLE_APB3_SLV_REG5_OFFSET 	20


    struct ctrl_reg {
    	unsigned int lfsr_stop	        :1;
    	unsigned int reserved			:31;
    
    }apb3_ctrl_reg;
    
    struct ctrl_reg2 {
		unsigned int mem_start	        :1;
		unsigned int rsv0               :7;
		unsigned int ilen               :8;
		unsigned int rsv1   			:16;

	}owrite_crtl;


    u32 apb3_read(u32 slave)
    {
    	return read_u32(slave+EXAMPLE_APB3_SLV_REG0_OFFSET);
    	
    }
    
    void apb3_ctrl_write(u32 slave, struct ctrl_reg *cfg)
    {
        write_u32(*(int *)cfg, slave+EXAMPLE_APB3_SLV_REG1_OFFSET);
    }
    
    void cfg_write(u32 slave, struct ctrl_reg2 *cfg)
	{
		write_u32(*(int *)cfg, slave+EXAMPLE_APB3_SLV_REG3_OFFSET);
	}

	void cfg_data(u32 slave, u32 data)
	{
		write_u32(data, slave+EXAMPLE_APB3_SLV_REG4_OFFSET);
	}

	void cfg_addr(u32 slave, u32  addr)
	{
		write_u32(addr, slave+EXAMPLE_APB3_SLV_REG5_OFFSET);
	}
    
    void apb3_ctrl_write(u32 slave, struct ctrl_reg *cfg);
    void cfg_write(u32 slave, struct ctrl_reg2 *cfg);
	void cfg_data(u32 slave, u32 data);
	void cfg_addr(u32 slave, u32 addr);
    u32 apb3_read(u32 slave);


